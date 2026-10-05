"""Run model-written Python away from the service process.

Some questions ("year-over-year growth", "market share per corporation") are
easier to express as a few lines of pandas than as SQL. That means executing
code a language model wrote, which is only acceptable if it cannot touch
anything.

An in-process sandbox built on a ``__builtins__`` denylist is **escapable** —
``().__class__.__base__.__subclasses__()`` walks straight to ``os`` — so the
code does not run in this process at all. It runs in a throwaway child:

* **environment allowlist** — an escape to ``os.environ`` finds nothing (an
  allowlist, not a denylist: a denylist leaks every secret whose name we forget
  to add);
* **wall-clock timeout** enforced by the parent, which kills the infinite loops
  an in-process guard cannot;
* **memory and CPU limits** inside the child, where the platform provides them;
* **an audit hook** that refuses network, process spawn and dynamic code
  loading at the interpreter level, below any Python-level trick;
* **JSON back, never pickle** — the child is the untrusted side, so the parent
  refuses to unpickle what it produced.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import settings

logger = logging.getLogger(__name__)

RUNNER = str(Path(__file__).with_name("_sandbox_runner.py"))

# Everything else — every secret, present or added later — is dropped.
ENV_ALLOWLIST = (
    "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR",
    "LANG", "LC_ALL", "LC_CTYPE", "TZ", "PYTHONPATH",
)


@dataclass
class SandboxResult:
    ok: bool
    value: object = None
    stdout: str = ""
    error: str = ""


def run(code: str, frame: pd.DataFrame) -> SandboxResult:
    """Execute ``code`` with ``df`` bound to ``frame``.

    The code is expected to leave its answer in a variable named ``result``.
    """
    with tempfile.TemporaryDirectory(prefix="p2s-sandbox-") as tmp:
        tmp_path = Path(tmp)
        data_file = tmp_path / "input.json"
        code_file = tmp_path / "code.py"
        out_file = tmp_path / "output.json"

        # JSON in as well as out: the child never unpickles us either.
        data_file.write_text(frame.to_json(orient="split"), encoding="utf-8")
        code_file.write_text(code, encoding="utf-8")

        env = {k: os.environ[k] for k in ENV_ALLOWLIST if k in os.environ}
        env["P2S_SANDBOX_MEMORY_MB"] = str(settings.sandbox_memory_mb)
        env["P2S_SANDBOX_TIMEOUT"] = str(settings.sandbox_timeout_s)
        # The import path is handed over explicitly rather than left to the
        # child's own site resolution: that resolution reads environment
        # variables (APPDATA, HOME) the allowlist deliberately drops, and a
        # sandbox whose dependencies load only when a secret-bearing variable
        # happens to be present is a sandbox nobody will keep enabled.
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)

        try:
            completed = subprocess.run(
                # No -I: it also drops the user site directory, and the child
                # needs pandas and its dependencies. Isolation comes from the
                # environment allowlist, the audit hook and the limits below —
                # not from an interpreter flag.
                [sys.executable, RUNNER, str(data_file), str(code_file), str(out_file)],
                env=env,
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=settings.sandbox_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(ok=False, error=f"timed out after {settings.sandbox_timeout_s}s")

        if completed.returncode != 0:
            detail = (completed.stderr or "").strip().splitlines()
            return SandboxResult(
                ok=False,
                error=detail[-1] if detail else f"exit code {completed.returncode}",
                stdout=completed.stdout,
            )

        if not out_file.exists():
            return SandboxResult(ok=False, error="child produced no result", stdout=completed.stdout)

        payload = json.loads(out_file.read_text(encoding="utf-8"))
        if not payload.get("ok"):
            return SandboxResult(ok=False, error=payload.get("error", "unknown error"))

        return SandboxResult(ok=True, value=payload.get("result"), stdout=payload.get("stdout", ""))
