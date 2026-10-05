"""The untrusted side of the sandbox. Never import this from the service.

Invoked as ``python -I _sandbox_runner.py <input.json> <code.py> <output.json>``
by :mod:`sandbox`. It installs its own restrictions before reading the code,
then runs it and writes a JSON result.

The audit hook is the part that matters: it sits below the Python level, so it
catches a socket opened through ``ctypes`` or an import resolved dynamically,
which a ``__builtins__`` denylist never sees.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout

BLOCKED_EVENTS = (
    "socket.", "urllib.", "ftplib.", "smtplib.",     # network
    "subprocess.", "os.exec", "os.fork", "os.spawn", "os.posix_spawn",  # processes
    "ctypes.", "cpython.",                            # native code
    "os.system", "pty.spawn", "shutil.",
)

ALLOWED_OPEN_PREFIXES: tuple[str, ...] = ()


def _install_audit_hook(workdir: str) -> None:
    import sys as _sys

    def hook(event: str, args):  # noqa: ANN001 - audit hook signature
        if event.startswith(BLOCKED_EVENTS):
            raise PermissionError(f"blocked by sandbox policy: {event}")
        if event == "open":
            path = os.path.abspath(str(args[0]))
            mode = args[1] or "r"
            # Reads stay inside the throwaway directory; writes are refused.
            if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
                raise PermissionError("blocked by sandbox policy: write")
            if not path.startswith(workdir) and not _is_stdlib(path):
                raise PermissionError(f"blocked by sandbox policy: read {path}")

    _sys.addaudithook(hook)


def _is_stdlib(path: str) -> bool:
    roots = [p for p in (sys.prefix, sys.base_prefix) if p]
    return any(os.path.abspath(path).startswith(os.path.abspath(r)) for r in roots)


def _apply_limits() -> None:
    """POSIX resource limits. A no-op on Windows, where they do not exist."""
    try:
        import resource
    except ImportError:
        return

    memory_mb = int(os.environ.get("P2S_SANDBOX_MEMORY_MB", "512"))
    cpu_s = int(os.environ.get("P2S_SANDBOX_TIMEOUT", "15"))
    for limit, value in (
        (resource.RLIMIT_AS, memory_mb * 1024 * 1024),
        (resource.RLIMIT_CPU, cpu_s),
        (resource.RLIMIT_NPROC, 0),
        (resource.RLIMIT_FSIZE, 0),
    ):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass


def _to_jsonable(value):
    """Shrink whatever the code produced into something JSON can carry."""
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        return {"type": "table", "columns": list(value.columns), "rows": value.head(200).values.tolist()}
    if isinstance(value, pd.Series):
        return {"type": "series", "index": [str(i) for i in value.index[:200]],
                "values": value.head(200).tolist()}
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> int:
    data_path, code_path, out_path = sys.argv[1:4]
    workdir = os.path.abspath(os.path.dirname(out_path))

    import pandas as pd  # imported before the hook: afterwards it would be blocked

    with open(data_path, encoding="utf-8") as handle:
        df = pd.read_json(io.StringIO(handle.read()), orient="split")
    with open(code_path, encoding="utf-8") as handle:
        code = handle.read()

    _apply_limits()
    _install_audit_hook(workdir)

    namespace: dict = {"df": df, "pd": pd, "result": None}
    buffer = io.StringIO()
    payload: dict

    try:
        with redirect_stdout(buffer):
            exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102 - the point of the file
        payload = {
            "ok": True,
            "result": _to_jsonable(namespace.get("result")),
            "stdout": buffer.getvalue()[:4000],
        }
    except BaseException as exc:  # noqa: BLE001 - everything is reported, nothing escapes
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # Written with the raw file descriptor: the audit hook refuses `open` for
    # writing, and it applies to this module too.
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, json.dumps(payload, default=str).encode("utf-8"))
    finally:
        os.close(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
