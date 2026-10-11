"""Forecasting models, NumPy only.

Four of them, and three are deliberately trivial. A forecasting project without
baselines cannot tell you whether its model is any good: "MAPE 14%" means
nothing until you know that answering "same month last year" scores 19% and
answering "last month's value" scores 23% on the same data. Most of the value
in a demand forecast is captured by the seasonal naive; the job of a real model
is to beat it, and the job of the harness is to prove whether it did.

Holt-Winters is the one real model here. It is the right level for monthly
pharmaceutical demand: five years of history is 60 points, which is too few to
fit anything with many parameters and plenty for a level/trend/season
decomposition. Parameters are chosen by a coarse grid search minimising
one-step in-sample error — robust, deterministic, and fast enough to backtest
hundreds of series without a model-fitting library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Model:
    """A model is fitted on a history and asked for the next ``horizon`` values."""

    name = "model"

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}>"


class Naive(Model):
    """Tomorrow looks like today. The floor every other model must clear."""

    name = "naive"

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        return np.repeat(history[-1], horizon)


class SeasonalNaive(Model):
    """The same month last year.

    On seasonal demand this is a genuinely strong competitor, and it is the
    baseline a planner is implicitly using when they say "last January we sold
    this much".
    """

    name = "seasonal_naive"

    def __init__(self, season: int = 12):
        self.season = season

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if history.size < self.season:
            return Naive().forecast(history, horizon)
        last_season = history[-self.season:]
        return np.array([last_season[i % self.season] for i in range(horizon)])


class MovingAverage(Model):
    """Average of the last ``window`` months — the rule of thumb in most ERPs."""

    name = "moving_average"

    def __init__(self, window: int = 6):
        self.window = window

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        window = min(self.window, history.size)
        return np.repeat(float(np.mean(history[-window:])), horizon)


class Drift(Model):
    """Last value, continued along the average slope of the history."""

    name = "drift"

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if history.size < 2:
            return Naive().forecast(history, horizon)
        slope = (history[-1] - history[0]) / (history.size - 1)
        return history[-1] + slope * np.arange(1, horizon + 1)


@dataclass
class _HWState:
    level: float
    trend: float
    season: np.ndarray
    sse: float


class HoltWinters(Model):
    """Triple exponential smoothing: level, trend and seasonality.

    Seasonality is multiplicative by default, which is what demand usually does
    — a winter peak is "20% above the level", not "+4 000 units", and the level
    moves over five years. It falls back to additive when the history contains
    zeros or near-zeros, where a multiplicative index is undefined.
    """

    name = "holt_winters"

    def __init__(
        self,
        season: int = 12,
        seasonal: str = "auto",          # auto | mul | add | none
        damped: bool = True,
        grid: int = 4,
    ):
        self.season = season
        self.seasonal = seasonal
        self.damped = damped
        self.grid = grid

    # -- fitting -----------------------------------------------------------

    def _seasonal_mode(self, history: np.ndarray) -> str:
        if self.seasonal != "auto":
            return self.seasonal
        if history.size < 2 * self.season:
            return "none"
        # A multiplicative index needs a level that never approaches zero.
        if float(np.min(history)) <= max(1e-6, 0.01 * float(np.mean(history))):
            return "add"
        return "mul"

    def _initial_state(self, history: np.ndarray, mode: str) -> tuple[float, float, np.ndarray]:
        season = self.season
        if mode == "none" or history.size < 2 * season:
            level = float(np.mean(history[: min(season, history.size)]))
            trend = 0.0
            indices = np.ones(season) if mode == "mul" else np.zeros(season)
            return level, trend, indices

        # Average of the first two full cycles, the classic initialisation.
        first = float(np.mean(history[:season]))
        second = float(np.mean(history[season: 2 * season]))
        level = first
        trend = (second - first) / season

        indices = np.empty(season)
        for i in range(season):
            points = history[i::season]
            if mode == "mul":
                indices[i] = float(np.mean(points)) / first if first else 1.0
            else:
                indices[i] = float(np.mean(points)) - first

        if mode == "mul":
            indices = np.where(indices <= 1e-6, 1.0, indices)
            indices *= season / float(np.sum(indices))
        else:
            indices -= float(np.mean(indices))

        return level, trend, indices

    def _run(
        self, history: np.ndarray, mode: str,
        alpha: float, beta: float, gamma: float, phi: float,
    ) -> _HWState:
        season = self.season
        level, trend, indices = self._initial_state(history, mode)
        indices = indices.copy()
        sse = 0.0

        for t, observed in enumerate(history):
            index = indices[t % season]

            if mode == "mul":
                predicted = (level + phi * trend) * index
            elif mode == "add":
                predicted = level + phi * trend + index
            else:
                predicted = level + phi * trend

            sse += (observed - predicted) ** 2

            previous_level = level
            if mode == "mul":
                safe = index if abs(index) > 1e-6 else 1.0
                level = alpha * (observed / safe) + (1 - alpha) * (level + phi * trend)
                indices[t % season] = (
                    gamma * (observed / level if abs(level) > 1e-6 else 1.0)
                    + (1 - gamma) * index
                )
            elif mode == "add":
                level = alpha * (observed - index) + (1 - alpha) * (level + phi * trend)
                indices[t % season] = gamma * (observed - level) + (1 - gamma) * index
            else:
                level = alpha * observed + (1 - alpha) * (level + phi * trend)

            trend = beta * (level - previous_level) + (1 - beta) * phi * trend

        return _HWState(level=level, trend=trend, season=indices, sse=sse)

    def _fit(self, history: np.ndarray, mode: str) -> tuple[_HWState, tuple[float, float, float, float]]:
        # A coarse grid, searched exhaustively: deterministic, dependency-free,
        # and on 60 monthly points the optimum is flat enough that a finer grid
        # buys nothing a backtest can measure.
        values = np.linspace(0.1, 0.9, self.grid)
        phis = (0.98, 0.85) if self.damped else (1.0,)
        gammas = values if mode != "none" else (0.0,)

        best_state: _HWState | None = None
        best_params = (0.3, 0.1, 0.1, 1.0)

        for alpha in values:
            for beta in values:
                for gamma in gammas:
                    for phi in phis:
                        state = self._run(history, mode, alpha, beta, gamma, phi)
                        if not np.isfinite(state.sse):
                            continue
                        if best_state is None or state.sse < best_state.sse:
                            best_state = state
                            best_params = (alpha, beta, gamma, phi)

        if best_state is None:  # pathological history; fall back to the mean
            level = float(np.mean(history))
            best_state = _HWState(level, 0.0, np.ones(self.season), float("inf"))

        return best_state, best_params

    # -- forecasting -------------------------------------------------------

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        if history.size < 3:
            return Naive().forecast(history, horizon)

        mode = self._seasonal_mode(history)
        state, (_, _, _, phi) = self._fit(history, mode)

        out = np.empty(horizon)
        for h in range(1, horizon + 1):
            # Damped trend: the slope decays instead of running away, which is
            # what keeps a 12-month horizon from forecasting into orbit.
            damping = sum(phi ** i for i in range(1, h + 1)) if phi != 1.0 else h
            index = state.season[(history.size + h - 1) % self.season]

            if mode == "mul":
                out[h - 1] = (state.level + damping * state.trend) * index
            elif mode == "add":
                out[h - 1] = state.level + damping * state.trend + index
            else:
                out[h - 1] = state.level + damping * state.trend

        # Demand is never negative; a model that says otherwise is reporting an
        # extrapolation artefact, not a forecast.
        return np.maximum(out, 0.0)


def default_models(season: int = 12) -> list[Model]:
    return [
        Naive(),
        SeasonalNaive(season=season),
        MovingAverage(window=6),
        Drift(),
        HoltWinters(season=season),
    ]
