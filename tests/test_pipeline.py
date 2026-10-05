"""End-to-end behaviour, including the part that refuses to answer."""

from __future__ import annotations

from pharma_text2sql import pipeline


def test_explain_does_not_execute():
    """Phase one understands and stops.

    Nothing in the explanation carries a result — that is the whole point of the
    gate, and it is cheap to assert.
    """
    explanation = pipeline.explain("total sales in France in 2024")
    assert "France" in str(explanation.resolved)
    assert not hasattr(explanation, "rows")
    assert "total" in explanation.interpretation.lower()


def test_execute_answers_a_simple_question():
    answer = pipeline.execute("total sales in France in 2024")
    assert answer.ok, answer.error
    assert answer.row_count == 1
    assert answer.sql.upper().startswith("SELECT")
    assert answer.narration


def test_execute_groups_and_ranks():
    answer = pipeline.execute("top 3 molecules by sales in France in 2024")
    assert answer.ok, answer.error
    assert answer.row_count == 3
    values = [row[-1] for row in answer.rows]
    assert values == sorted(values, reverse=True)


def test_unknown_entity_asks_instead_of_answering():
    """The failure mode that matters.

    Without resolution this returns a confident total over every molecule in
    Spain — a wrong number that looks exactly like a right one.
    """
    answer = pipeline.execute("sales of NOTAMOLECULE in Spain")
    assert not answer.ok
    assert "NOTAMOLECULE" in answer.error


def test_ambiguous_entity_asks_which_one():
    answer = pipeline.execute("total sales in United")
    assert not answer.ok
    assert "United Kingdom" in answer.error and "United States" in answer.error


def test_a_stacked_drop_cannot_get_through():
    pipeline.execute("total sales for OMEZARIN; DROP TABLE sales")
    # Whatever happened above, the table is still there.
    follow_up = pipeline.execute("total sales in France in 2024")
    assert follow_up.ok, follow_up.error


def test_every_answer_carries_the_sql_that_produced_it():
    """Traceability: an answer nobody can audit is not an answer."""
    answer = pipeline.execute("total units by country in 2023")
    assert answer.ok, answer.error
    assert "FROM sales" in answer.sql
    assert "2023" in answer.sql
