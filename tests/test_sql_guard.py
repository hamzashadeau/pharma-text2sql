"""The guard is the safety net. These tests are the reason it can be trusted."""

from __future__ import annotations

import pytest

from pharma_text2sql import sql_guard


def test_plain_select_is_accepted():
    result = sql_guard.validate("SELECT country, sum(units) FROM sales GROUP BY country")
    assert result.ok
    assert "SELECT" in result.sql


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales",
        "DELETE FROM sales",
        "UPDATE sales SET units = 0",
        "INSERT INTO sales VALUES (1)",
        "CREATE TABLE evil (x INT)",
        "ALTER TABLE sales ADD COLUMN x INT",
    ],
)
def test_writes_are_refused(sql):
    assert not sql_guard.validate(sql).ok


def test_stacked_statement_is_refused():
    # The classic: a harmless SELECT with a second statement glued on.
    result = sql_guard.validate("SELECT 1 FROM sales; DROP TABLE sales")
    assert not result.ok
    assert "one statement" in result.error


def test_file_reading_functions_are_refused():
    result = sql_guard.validate("SELECT * FROM read_csv('/etc/passwd')")
    assert not result.ok


def test_unknown_table_is_refused():
    result = sql_guard.validate("SELECT * FROM secrets")
    assert not result.ok
    assert "unknown table" in result.error


def test_unknown_column_is_refused():
    result = sql_guard.validate("SELECT no_such_column FROM sales")
    assert not result.ok
    assert "unknown column" in result.error


def test_literal_containing_a_keyword_is_accepted():
    """A value is not an instruction.

    "EYE DROP" products and molecules like AMINODIHYDROPHTHALASINDIONE contain
    forbidden keywords inside *data*. A regex denylist refuses these perfectly
    legitimate questions; working on the parse tree does not.
    """
    result = sql_guard.validate(
        "SELECT sum(units) FROM sales WHERE product = 'Eye DROP solution'"
    )
    assert result.ok, result.error


def test_limit_is_forced_when_absent():
    result = sql_guard.validate("SELECT * FROM sales")
    assert result.ok
    assert "LIMIT" in result.sql.upper()


def test_oversized_limit_is_capped():
    result = sql_guard.validate("SELECT * FROM sales LIMIT 10000000")
    assert result.ok
    assert "10000000" not in result.sql


def test_empty_query_is_refused():
    assert not sql_guard.validate("   ").ok


def test_garbage_is_refused_without_raising():
    result = sql_guard.validate("this is not sql at all ((((")
    assert not result.ok
