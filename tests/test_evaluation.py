"""Scoring: faithfulness, run aggregates, run-to-run comparison, judge."""

from __future__ import annotations

import pandas as pd
import pytest

from pharma_text2sql.evaluation import compare as compare_runs
from pharma_text2sql.evaluation import faithfulness, judge
from pharma_text2sql.evaluation.run import QuestionResult, Run

FRAME = pd.DataFrame({
    "country": ["France", "Spain", "Japan"],
    "total_sales_eur": [37_680_097.44, 12_500_000.0, 9_000_000.0],
})


# ------------------------------------------------------------ faithfulness

def test_an_exact_figure_is_supported():
    result = faithfulness.check("France leads with 37,680,097.44 in sales.", FRAME)
    assert result.ok
    assert result.score == 1.0


def test_a_rounded_figure_is_supported():
    """Narration rounds. "37.7 million" is a correct statement, not a lie."""
    result = faithfulness.check("France leads with 37.7 million in sales.", FRAME)
    assert result.ok, result.summary()


def test_an_invented_figure_is_caught():
    """The failure this module exists for.

    The query ran, the table is right, and the sentence underneath states a
    number that is nowhere in it. No status counter can see this.
    """
    result = faithfulness.check("France leads with 54.2 million in sales.", FRAME)
    assert not result.ok
    assert any("54.2" in c.text for c in result.unsupported)


def test_the_row_count_is_supported():
    result = faithfulness.check("3 countries are listed.", FRAME)
    assert result.ok


def test_a_column_total_is_supported():
    """Narration routinely totals a column the table shows broken down."""
    total = FRAME["total_sales_eur"].sum()
    result = faithfulness.check(f"Together they sold {total:,.2f}.", FRAME)
    assert result.ok, result.summary()


def test_a_column_average_is_supported():
    """An analytics answer states derived aggregates constantly.

    Flagging "on average 19.7 million" as invented, when it is the mean of the
    column displayed right above it, is a false alarm — and false alarms are
    how a check ends up switched off.
    """
    mean = FRAME["total_sales_eur"].mean()
    result = faithfulness.check(f"On average {mean:,.2f} per country.", FRAME)
    assert result.ok, result.summary()


def test_a_column_maximum_is_supported():
    peak = FRAME["total_sales_eur"].max()
    result = faithfulness.check(f"The peak is {peak:,.2f}.", FRAME)
    assert result.ok, result.summary()


def test_widening_to_aggregates_still_catches_an_invented_figure():
    """The widening must not swallow the case the check exists for."""
    result = faithfulness.check("Sales reached 123,456,789 across the board.", FRAME)
    assert not result.ok


def test_digits_inside_a_label_are_not_figures():
    """"Aciclovex 20 MG" carries a 20 that claims nothing about a quantity.

    Flagging it would fill the report with noise, and a check that cries wolf
    is one people stop reading.
    """
    frame = pd.DataFrame({"product": ["Aciclovex 20 MG"], "units": [500]})
    result = faithfulness.check("Leading product: Aciclovex 20 MG (units 500).", frame)
    assert result.ok, result.summary()


def test_period_labels_are_not_figures():
    frame = pd.DataFrame({"period": ["2024-01", "2024-02"], "units": [10, 20]})
    result = faithfulness.check("Sales peaked in 2024-01 with 10 units.", frame)
    assert result.ok, result.summary()


def test_percentages_are_reported_as_unchecked_not_as_supported():
    """A share is derived; verifying it would need the derivation.

    Counting it as supported would be a lie of convenience, and counting it as
    unsupported would be a false alarm. It gets its own bucket.
    """
    result = faithfulness.check("France represents 63% of the total.", FRAME)
    assert result.ok
    assert len(result.unchecked) == 1
    assert result.unchecked[0].is_percentage


def test_numbers_restated_from_the_question_are_context_not_claims():
    result = faithfulness.check(
        "In 2024, France led the ranking.", FRAME, question="sales by country in 2024"
    )
    assert result.ok
    assert result.checked == 0


def test_a_narration_without_figures_is_faithful():
    result = faithfulness.check("France leads the ranking.", FRAME)
    assert result.ok
    assert result.score == 1.0


def test_european_number_formats_are_read_correctly():
    assert faithfulness._to_float("1 234 567") == pytest.approx(1234567)
    assert faithfulness._to_float("1.234.567,89") == pytest.approx(1234567.89)
    assert faithfulness._to_float("1,234,567.89") == pytest.approx(1234567.89)
    assert faithfulness._to_float("12,5") == pytest.approx(12.5)


def test_scale_suffixes_are_applied():
    cited = faithfulness.citations("4.2 million units and 3 bn euros")
    assert cited[0].value == pytest.approx(4_200_000)
    assert cited[1].value == pytest.approx(3_000_000_000)


# --------------------------------------------------------------------- run

def _result(id_: str, **kwargs) -> QuestionResult:
    base = {"question": f"q {id_}", "level": "1", "status": "success"}
    base.update(kwargs)
    return QuestionResult(id=id_, **base)


def test_accuracy_is_none_without_ground_truth_not_zero():
    """The distinction that keeps an evaluation honest.

    An unlabelled run reported as 0% accurate is a false statement that reads
    like a measurement.
    """
    run = Run(results=[_result("a"), _result("b")])
    assert run.accuracy() is None
    assert "not measured" in run.report()


def test_accuracy_counts_only_labelled_questions():
    run = Run(results=[
        _result("a", passed=True), _result("b", passed=False), _result("c"),
    ])
    assert run.accuracy() == pytest.approx(0.5)


def test_latency_is_reported_at_percentiles():
    run = Run(results=[_result(str(i), duration_s=float(i)) for i in range(1, 101)])
    latency = run.latency()
    assert latency["p50"] == pytest.approx(50.5)
    assert latency["p95"] == pytest.approx(95.05, abs=0.5)
    assert latency["max"] == 100


def test_cost_uses_the_price_table():
    run = Run(results=[_result("a", model="m", tokens_in=1_000_000, tokens_out=500_000)])
    cost = run.cost(prices={"m": (2.0, 6.0)})
    assert cost["total"] == pytest.approx(2.0 + 3.0)


def test_an_unknown_model_costs_zero_rather_than_guessing():
    run = Run(results=[_result("a", model="unlisted", tokens_in=10_000)])
    assert run.cost()["total"] == 0.0


def test_a_run_survives_a_round_trip_through_json(tmp_path):
    run = Run(label="x", mode="offline", results=[_result("a", passed=True, duration_s=1.5)])
    path = tmp_path / "run.json"
    run.to_json(path)

    loaded = Run.from_json(path)
    assert loaded.label == "x"
    assert loaded.results[0].passed is True
    assert loaded.results[0].duration_s == 1.5


# ----------------------------------------------------------------- compare

def test_a_question_that_stops_passing_is_a_regression():
    before = Run(label="base", results=[_result("a", passed=True)])
    after = Run(label="new", results=[_result("a", passed=False)])

    comparison = compare_runs.compare(before, after)
    assert not comparison.ok
    assert comparison.regressions[0].id == "a"


def test_a_question_that_starts_passing_is_an_improvement():
    before = Run(results=[_result("a", passed=False)])
    after = Run(results=[_result("a", passed=True)])

    comparison = compare_runs.compare(before, after)
    assert comparison.ok
    assert len(comparison.improvements) == 1


def test_regressions_are_detected_without_any_ground_truth():
    """The case that makes a frozen baseline useful.

    Nobody wrote down the right answer; an answer that worked yesterday and
    errors today is still a regression.
    """
    before = Run(results=[_result("a", status="success")])
    after = Run(results=[_result("a", status="error")])

    comparison = compare_runs.compare(before, after)
    assert not comparison.ok
    assert "success" in comparison.regressions[0].detail


def test_a_drop_in_faithfulness_is_a_regression():
    """Status unchanged, answer still arrives — and now cites a figure that
    is not in the table. Invisible to a status counter."""
    before = Run(results=[_result("a", faithfulness=1.0)])
    after = Run(results=[_result("a", faithfulness=0.5)])

    comparison = compare_runs.compare(before, after)
    assert not comparison.ok
    assert "faithfulness" in comparison.regressions[0].detail


def test_latency_is_reported_but_never_counted_as_a_regression():
    """Failing a build on one slow provider call teaches people to ignore
    the build."""
    before = Run(results=[_result("a", duration_s=1.0)])
    after = Run(results=[_result("a", duration_s=10.0)])

    comparison = compare_runs.compare(before, after)
    assert comparison.ok
    assert len(comparison.latency_changes) == 1


def test_a_changed_question_set_is_surfaced():
    before = Run(results=[_result("a"), _result("b")])
    after = Run(results=[_result("b"), _result("c")])

    comparison = compare_runs.compare(before, after)
    assert comparison.only_in_baseline == ["a"]
    assert comparison.only_in_candidate == ["c"]


def test_identical_runs_report_no_change():
    run = Run(results=[_result("a", passed=True)])
    comparison = compare_runs.compare(run, run)
    assert comparison.ok
    assert "no change" in comparison.report()


# ------------------------------------------------------------------- judge

def test_the_judge_reports_unavailable_rather_than_passing_silently(monkeypatch):
    """A judge that returns "pass" when it did not run is worse than no judge."""
    monkeypatch.setattr(judge.llm, "available", lambda: False)

    verdict = judge.judge("question", "answer")
    assert verdict.verdict == judge.UNAVAILABLE
    assert not verdict.passed
    assert not verdict.available


def test_the_judge_summary_says_it_did_not_run(monkeypatch):
    monkeypatch.setattr(judge.llm, "available", lambda: False)
    verdicts = judge.judge_all([{"question": "q", "answer": "a"}])
    assert "not run" in judge.summarise(verdicts)


def test_an_unparseable_verdict_is_unavailable_not_a_pass(monkeypatch):
    monkeypatch.setattr(judge.llm, "available", lambda: True)
    monkeypatch.setattr(judge.llm, "complete_json", lambda *a, **k: {"verdict": "maybe"})

    verdict = judge.judge("question", "answer")
    assert verdict.verdict == judge.UNAVAILABLE


def test_a_judge_failure_never_breaks_the_run(monkeypatch):
    monkeypatch.setattr(judge.llm, "available", lambda: True)

    def explode(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(judge.llm, "complete_json", explode)
    verdict = judge.judge("question", "answer")
    assert verdict.verdict == judge.UNAVAILABLE
    assert "provider down" in verdict.reason
