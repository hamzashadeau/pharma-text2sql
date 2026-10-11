"""Forecast error metrics, and which one to believe.

MAPE is what a business asks for and the first one that misleads them. It
divides by the actual, so a month where a product sold 3 units contributes more
error than a month where it sold 30 000 — and pharmaceutical sales are full of
small months: launches, stockouts, discontinued presentations, hospital
products ordered in batches. A MAPE computed over a portfolio is dominated by
its smallest series.

So MAPE is reported because it is the number people expect, next to two that
survive the same data:

* **WAPE** — total absolute error over total volume. Weights each month by how
  much it actually sold, which is what a supply chain cares about.
* **MASE** — error scaled by the seasonal-naive error on the training window.
  Below 1 the model beats "same month last year"; above 1 it does not. It is
  the only one here that is comparable across products of different sizes, and
  the only one that answers "is this model worth anything at all".

Bias is reported separately because it is actionable in a way accuracy is not:
a model that is 10% wrong in both directions costs safety stock, a model that
is 10% low every month causes stockouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPSILON = 1e-9


@dataclass(frozen=True)
class Scores:
    mape: float
    wape: float
    mase: float
    mae: float
    rmse: float
    bias: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {
            "mape": self.mape, "wape": self.wape, "mase": self.mase,
            "mae": self.mae, "rmse": self.rmse, "bias": self.bias, "n": self.n,
        }


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, over non-zero actuals only.

    Months with zero sales are dropped rather than given an infinite error.
    That is the conventional treatment, and it is also why MAPE flatters a
    model on intermittent series: the hardest months are the ones excluded.
    """
    mask = np.abs(actual) > EPSILON
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted absolute percentage error: total error over total volume."""
    denominator = np.sum(np.abs(actual))
    if denominator < EPSILON:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / denominator * 100)


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Signed error as a share of volume. Negative means under-forecasting."""
    denominator = np.sum(np.abs(actual))
    if denominator < EPSILON:
        return float("nan")
    return float(np.sum(predicted - actual) / denominator * 100)


def mase(actual: np.ndarray, predicted: np.ndarray, train: np.ndarray, season: int = 12) -> float:
    """Mean absolute scaled error.

    The scale is the in-sample seasonal-naive MAE: the error you would make by
    answering "the same month last year" on the training window. Computing the
    scale on the *training* data, never on the test window, is what keeps it
    honest — scaling by the test period would let a model look good simply by
    being evaluated on a quiet stretch.
    """
    if train.size <= season:
        season = 1
    if train.size <= season:
        return float("nan")

    naive_errors = np.abs(train[season:] - train[:-season])
    scale = float(np.mean(naive_errors))
    if scale < EPSILON:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / scale)


def score(
    actual: np.ndarray,
    predicted: np.ndarray,
    train: np.ndarray | None = None,
    season: int = 12,
) -> Scores:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")

    scaled = (
        mase(actual, predicted, np.asarray(train, dtype=float), season)
        if train is not None and len(train)
        else float("nan")
    )

    return Scores(
        mape=mape(actual, predicted),
        wape=wape(actual, predicted),
        mase=scaled,
        mae=mae(actual, predicted),
        rmse=rmse(actual, predicted),
        bias=bias(actual, predicted),
        n=int(actual.size),
    )
