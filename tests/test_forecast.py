"""Forecasting: the metrics, the models, and the one property that matters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharma_text2sql.forecast import backtest as bt
from pharma_text2sql.forecast import metrics
from pharma_text2sql.forecast.models import (
    Drift,
    HoltWinters,
    Model,
    MovingAverage,
    Naive,
    SeasonalNaive,
)


# ---------------------------------------------------------------- metrics

def test_perfect_forecast_scores_zero():
    actual = np.array([10.0, 20.0, 30.0])
    scores = metrics.score(actual, actual.copy(), train=actual)
    assert scores.mape == pytest.approx(0)
    assert scores.wape == pytest.approx(0)
    assert scores.mae == pytest.approx(0)
    assert scores.bias == pytest.approx(0)


def test_mape_ignores_zero_months_and_wape_does_not():
    """The reason both are reported.

    One zero month with a non-zero forecast is invisible to MAPE and plainly
    visible to WAPE. On intermittent pharmaceutical demand that gap is the
    difference between a reassuring number and a true one.
    """
    actual = np.array([0.0, 100.0])
    predicted = np.array([50.0, 100.0])

    assert metrics.mape(actual, predicted) == pytest.approx(0)
    assert metrics.wape(actual, predicted) == pytest.approx(50)


def test_bias_sign_distinguishes_over_from_under_forecasting():
    actual = np.array([100.0, 100.0])
    assert metrics.bias(actual, np.array([90.0, 90.0])) < 0     # under-forecast
    assert metrics.bias(actual, np.array([110.0, 110.0])) > 0   # over-forecast

    # Errors that cancel out: accurate on average, wrong every month. Bias says
    # nothing is wrong, which is exactly why accuracy is reported next to it.
    assert metrics.bias(actual, np.array([80.0, 120.0])) == pytest.approx(0)
    assert metrics.wape(actual, np.array([80.0, 120.0])) == pytest.approx(20)


def test_mase_is_one_for_a_seasonal_naive_forecast():
    rng = np.random.default_rng(0)
    train = np.tile(np.arange(1, 13, dtype=float), 3) + rng.normal(0, 0.1, 36)

    actual = train[-12:]
    predicted = train[-24:-12]
    scaled = metrics.mase(actual, predicted, train, season=12)

    assert 0.5 < scaled < 2.0


def test_scores_reject_mismatched_shapes():
    with pytest.raises(ValueError):
        metrics.score(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------- models

def test_naive_repeats_the_last_value():
    assert Naive().forecast(np.array([1.0, 2.0, 7.0]), 3).tolist() == [7.0, 7.0, 7.0]


def test_seasonal_naive_repeats_the_cycle():
    history = np.arange(1, 25, dtype=float)
    predicted = SeasonalNaive(season=12).forecast(history, 3)
    assert predicted.tolist() == [13.0, 14.0, 15.0]


def test_seasonal_naive_falls_back_when_history_is_shorter_than_a_cycle():
    predicted = SeasonalNaive(season=12).forecast(np.array([5.0, 6.0]), 2)
    assert predicted.tolist() == [6.0, 6.0]


def test_moving_average_uses_the_window():
    predicted = MovingAverage(window=3).forecast(np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]), 1)
    assert predicted[0] == pytest.approx(20.0)


def test_drift_follows_the_slope():
    predicted = Drift().forecast(np.array([10.0, 12.0, 14.0]), 2)
    assert predicted.tolist() == pytest.approx([16.0, 18.0])


def test_holt_winters_recovers_a_clean_seasonal_series():
    months = np.arange(60)
    seasonal = 1 + 0.3 * np.cos(months % 12 * np.pi / 6)
    history = 1000 * 1.01 ** months * seasonal

    predicted = HoltWinters(season=12).forecast(history[:48], 12)
    actual = history[48:60]

    assert metrics.wape(actual, predicted) < 10


def test_forecasts_are_never_negative():
    """A declining series extrapolated far enough goes below zero.

    Demand does not. Returning a negative forecast would propagate into a
    negative replenishment and look like a data problem three systems away.
    """
    history = np.linspace(500, 10, 36)
    predicted = HoltWinters(season=12).forecast(history, 24)
    assert (predicted >= 0).all()


def test_models_cope_with_a_very_short_history():
    for model in (Naive(), SeasonalNaive(), MovingAverage(), Drift(), HoltWinters()):
        predicted = model.forecast(np.array([42.0]), 2)
        assert len(predicted) == 2
        assert np.isfinite(predicted).all()


# ---------------------------------------------------------------- backtest

def _frame(periods: list[str], values: list[float], series: str = "A") -> pd.DataFrame:
    return pd.DataFrame({"series": series, "period": periods, "value": values})


def test_to_series_fills_missing_months_with_zero():
    """A month absent from a sales extract means nothing sold, not unknown.

    Left as a gap it silently shortens the cycle and shifts every lag by one —
    which surfaces much later as an inexplicably poor seasonal model.
    """
    frame = _frame(["2024-01", "2024-03"], [10.0, 30.0])
    series = bt.to_series(frame, series_col="series", date_col="period", value_col="value")

    assert series["A"].tolist() == [10.0, 0.0, 30.0]
    assert list(series["A"].index) == ["2024-01", "2024-02", "2024-03"]


def test_series_too_short_is_skipped_not_scored():
    periods = [f"2024-{m:02d}" for m in range(1, 7)]
    series = bt.to_series(_frame(periods, [1.0] * 6),
                          series_col="series", date_col="period", value_col="value")

    result = bt.run(series, config=bt.BacktestConfig(min_train=24, horizon=3))
    assert result.per_series == []
    assert "A" in result.skipped


def test_the_backtest_never_shows_a_model_the_future():
    """The property the whole harness exists to guarantee.

    A model that sees even one future point can score beautifully and forecast
    nothing. This spy records the history it is handed at every origin and
    asserts it is exactly the window before that origin.
    """
    observed: list[int] = []
    horizon, min_train = 3, 24
    values = list(np.arange(1, 49, dtype=float))
    periods = [f"{2021 + i // 12}-{i % 12 + 1:02d}" for i in range(48)]

    class Spy(Model):
        name = "spy"

        def forecast(self, history, h):
            observed.append(history.size)
            # Any future leak would show up as a history longer than the origin.
            assert history.tolist() == values[: history.size]
            return np.repeat(history[-1], h)

    series = bt.to_series(_frame(periods, values),
                          series_col="series", date_col="period", value_col="value")
    bt.run(series, models=[Spy()], config=bt.BacktestConfig(horizon=horizon, min_train=min_train))

    expected_origins = list(range(min_train, len(values) - horizon + 1))
    assert observed == expected_origins
    assert min(observed) == min_train
    assert max(observed) == len(values) - horizon


def test_summary_reports_one_row_per_model_and_a_win_rate():
    rng = np.random.default_rng(1)
    months = np.arange(60)
    rows = []
    for key in ("A", "B"):
        seasonal = 1 + 0.25 * np.cos(months % 12 * np.pi / 6)
        values = 500 * 1.005 ** months * seasonal * rng.normal(1, 0.05, 60)
        for i, value in enumerate(values):
            rows.append({
                "series": key,
                "period": f"{2021 + i // 12}-{i % 12 + 1:02d}",
                "value": float(value),
            })

    series = bt.to_series(pd.DataFrame(rows),
                          series_col="series", date_col="period", value_col="value")
    result = bt.run(series, models=[Naive(), SeasonalNaive()],
                    config=bt.BacktestConfig(horizon=3, min_train=36))

    summary = result.summary()
    assert set(summary["model"]) == {"naive", "seasonal_naive"}
    assert (summary["series"] == 2).all()
    # The baseline cannot beat itself.
    assert summary.loc[summary["model"] == "seasonal_naive", "beats_seasonal_naive"].iloc[0] == 0
