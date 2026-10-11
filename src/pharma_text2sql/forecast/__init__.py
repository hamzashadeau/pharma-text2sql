"""Demand forecasting and rolling-origin backtesting.

Independent of the text-to-SQL pipeline: it takes a long table of
``(series, date, value)`` and reports how well each model would have done,
measured the way a planner would have used it.

    from pharma_text2sql.forecast import backtest, models

    series = backtest.to_series(frame, series_col="product",
                                date_col="period", value_col="units")
    result = backtest.run(series)
    print(result.summary())
"""

from . import backtest, metrics, models  # noqa: F401

__all__ = ["backtest", "metrics", "models"]
