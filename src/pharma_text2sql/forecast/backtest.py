"""Rolling-origin backtest.

The only honest way to state a forecast error is to compute it the way the
forecast would have been used: standing at the end of month *t*, knowing
nothing after it, predicting *t+1 … t+h*. Then move one month forward and do
it again.

Two mistakes this is built to avoid, both of which produce excellent numbers
and worthless models:

* **Fitting on the whole history, then measuring on part of it.** The model has
  already seen the answer. In-sample error is not a forecast error.
* **One split.** "Trained to 2024, tested on 2025" is a single draw. If 2025
  was calm the number flatters the model; if a product was discontinued it
  damns it. A rolling origin averages over every month of the test period, and
  the spread across origins is itself information.

The harness is deliberately agnostic about where the data comes from: it takes
a long table of ``(series, date, value)``. The demo points it at the synthetic
sales dataset; a different caller can point it at a CSV export of real history
without this module knowing the difference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics
from .models import Model, default_models

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    horizon: int = 3            # months ahead a planner actually commits to
    min_train: int = 24         # two full cycles before the first forecast
    season: int = 12
    step: int = 1               # months between consecutive origins
    max_series: int | None = None


@dataclass
class SeriesResult:
    series: str
    model: str
    scores: metrics.Scores
    origins: int
    volume: float


@dataclass
class BacktestResult:
    config: BacktestConfig
    per_series: list[SeriesResult] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """One row per model, aggregated across series.

        WAPE and bias are pooled by volume rather than averaged across series:
        averaging per-series percentages gives a tiny product the same say as
        the portfolio's biggest seller, which is how a forecast can be reported
        as accurate while the supply chain is on fire.
        """
        rows = []
        for model in sorted({r.model for r in self.per_series}):
            group = [r for r in self.per_series if r.model == model]
            weights = np.array([r.volume for r in group], dtype=float)
            total = weights.sum()

            def pooled(attribute: str) -> float:
                values = np.array([getattr(r.scores, attribute) for r in group], dtype=float)
                mask = np.isfinite(values)
                if not mask.any() or total <= 0:
                    return float("nan")
                return float(np.average(values[mask], weights=weights[mask]))

            def plain(attribute: str) -> float:
                values = np.array([getattr(r.scores, attribute) for r in group], dtype=float)
                values = values[np.isfinite(values)]
                return float(np.mean(values)) if values.size else float("nan")

            rows.append({
                "model": model,
                "series": len(group),
                "mape": plain("mape"),
                "wape": pooled("wape"),
                "mase": plain("mase"),
                "bias": pooled("bias"),
                "beats_seasonal_naive": float("nan"),
            })

        frame = pd.DataFrame(rows)
        return self._add_win_rate(frame)

    def _add_win_rate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Share of series where a model's MASE beats the seasonal naive.

        An average hides the shape of the win: a model can have a better mean
        error while losing on most products, carried by a handful of big ones.
        """
        baseline = {
            r.series: r.scores.mase
            for r in self.per_series
            if r.model == "seasonal_naive"
        }
        if not baseline:
            return frame

        rates = []
        for model in frame["model"]:
            group = [r for r in self.per_series if r.model == model]
            comparable = [
                r for r in group
                if np.isfinite(r.scores.mase) and np.isfinite(baseline.get(r.series, np.nan))
            ]
            if not comparable:
                rates.append(float("nan"))
                continue
            wins = sum(1 for r in comparable if r.scores.mase < baseline[r.series])
            rates.append(wins / len(comparable) * 100)

        frame["beats_seasonal_naive"] = rates
        return frame


def to_series(
    frame: pd.DataFrame,
    *,
    series_col: str,
    date_col: str,
    value_col: str,
) -> dict[str, pd.Series]:
    """Long table → one sorted monthly series per key, gaps filled with zeros.

    A missing month in a sales extract means "nothing was sold", not "unknown".
    Leaving the gap would silently shorten the seasonal cycle and shift every
    lag by one, which is the kind of error that shows up as a mysteriously bad
    seasonal model rather than as an exception.
    """
    out: dict[str, pd.Series] = {}
    for key, group in frame.groupby(series_col):
        series = (
            group.groupby(date_col)[value_col].sum().sort_index().astype(float)
        )
        index = pd.period_range(
            pd.Period(series.index.min(), freq="M"),
            pd.Period(series.index.max(), freq="M"),
            freq="M",
        ).astype(str)
        out[str(key)] = series.reindex(index, fill_value=0.0)
    return out


def run(
    series_by_key: dict[str, pd.Series],
    models: list[Model] | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    models = models or default_models(season=config.season)
    result = BacktestResult(config=config)

    keys = list(series_by_key)
    if config.max_series:
        # Largest series first: if coverage is capped, cap it on the products
        # that matter, and say so rather than sampling silently.
        keys = sorted(keys, key=lambda k: -float(series_by_key[k].sum()))[: config.max_series]
        if len(series_by_key) > len(keys):
            logger.info("limited to the %d largest series of %d", len(keys), len(series_by_key))

    for key in keys:
        values = series_by_key[key].to_numpy(dtype=float)
        needed = config.min_train + config.horizon
        if values.size < needed:
            result.skipped[key] = f"{values.size} months, needs {needed}"
            continue

        origins = list(range(config.min_train, values.size - config.horizon + 1, config.step))
        if not origins:
            result.skipped[key] = "no usable origin"
            continue

        for model in models:
            actuals: list[float] = []
            predictions: list[float] = []

            for origin in origins:
                train = values[:origin]
                actual = values[origin: origin + config.horizon]
                try:
                    predicted = model.forecast(train, config.horizon)
                except Exception:
                    logger.warning("%s failed on %s at origin %d", model.name, key, origin,
                                   exc_info=True)
                    continue
                actuals.extend(actual.tolist())
                predictions.extend(np.asarray(predicted, dtype=float).tolist())

            if not actuals:
                continue

            scores = metrics.score(
                np.array(actuals),
                np.array(predictions),
                train=values[: config.min_train],
                season=config.season,
            )
            result.per_series.append(
                SeriesResult(
                    series=key,
                    model=model.name,
                    scores=scores,
                    origins=len(origins),
                    volume=float(np.sum(np.abs(actuals))),
                )
            )

    return result
