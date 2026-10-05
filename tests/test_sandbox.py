"""The sandbox runs code a model wrote. These tests state what it may not do."""

from __future__ import annotations

import pandas as pd
import pytest

from pharma_text2sql import sandbox

FRAME = pd.DataFrame({"country": ["France", "Spain", "France"], "units": [10, 20, 30]})


def test_normal_code_returns_a_result():
    result = sandbox.run("result = int(df['units'].sum())", FRAME)
    assert result.ok, result.error
    assert result.value == 60


def test_dataframe_result_comes_back_as_a_table():
    result = sandbox.run("result = df.groupby('country')['units'].sum().reset_index()", FRAME)
    assert result.ok, result.error
    assert result.value["type"] == "table"
    assert "country" in result.value["columns"]


def test_network_access_is_blocked():
    code = "import socket; socket.socket().connect(('example.com', 80)); result = 'reached'"
    result = sandbox.run(code, FRAME)
    assert not result.ok
    assert result.value != "reached"


def test_process_spawn_is_blocked():
    code = "import subprocess; subprocess.run(['echo', 'hi']); result = 'spawned'"
    result = sandbox.run(code, FRAME)
    assert not result.ok


def test_writing_a_file_is_blocked():
    code = "open('escaped.txt', 'w').write('x'); result = 'wrote'"
    result = sandbox.run(code, FRAME)
    assert not result.ok


def test_reading_outside_the_sandbox_is_blocked():
    code = "result = open(r'C:\\Windows\\win.ini').read()"
    result = sandbox.run(code, FRAME)
    assert not result.ok


def test_the_classic_subclasses_escape_does_not_help():
    """``().__class__.__base__.__subclasses__()`` is why an in-process sandbox
    built on a builtins denylist is not a sandbox. Here it may well find the
    class — and still cannot do anything with it, because the restrictions are
    in the child process and in the interpreter's audit hook, not in a dict of
    allowed names."""
    code = (
        "ks = [c for c in ().__class__.__base__.__subclasses__() "
        "if c.__name__ == 'Popen']\n"
        "result = ks[0](['echo', 'hi']) if ks else 'no Popen'"
    )
    result = sandbox.run(code, FRAME)
    assert not result.ok or result.value == "no Popen"


def test_a_crash_is_reported_not_raised():
    result = sandbox.run("result = 1 / 0", FRAME)
    assert not result.ok
    assert "ZeroDivisionError" in result.error


@pytest.mark.slow
def test_an_infinite_loop_is_killed():
    result = sandbox.run("while True:\n    pass", FRAME)
    assert not result.ok
    assert "timed out" in result.error
