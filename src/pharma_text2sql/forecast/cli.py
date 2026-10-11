"""Run the backtest from the command line.

    python -m pharma_text2sql.forecast                      # synthetic dataset
    python -m pharma_text2sql.forecast --level molecule --horizon 6
    python -m pharma_text2sql.forecast --csv history.csv \
        --series-col product_code --date-col month --value-col units

The ``--csv`` form is the one that matters outside the demo: point it at an
export of real history and the same harness, the same metrics and the same
rolling origin apply, with nothing in this repository needing to know what the
source system looks like.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .. import schema
from . import backtest as bt
from .models import HoltWinters, default_models


def _load_demo(level: str, country: str | None) -> pd.DataFrame:
    where = f"WHERE country = '{country}'" if country else ""
    return schema.query(
        f'SELECT "{level}" AS series, period, sum(units) AS value '
        f"FROM {schema.settings.table_name} {where} "
        f'GROUP BY "{level}", period'
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pharma_text2sql.forecast")
    parser.add_argument("--csv", help="long table with one row per series and period")
    parser.add_argument("--series-col", default="series")
    parser.add_argument("--date-col", default="period")
    parser.add_argument("--value-col", default="value")
    parser.add_argument("--level", default="product",
                        help="demo only: column to forecast at (product, molecule, …)")
    parser.add_argument("--country", default=None, help="demo only: restrict to one country")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--min-train", type=int, default=24)
    parser.add_argument("--season", type=int, default=12)
    parser.add_argument("--max-series", type=int, default=40)
    parser.add_argument("--fast", action="store_true",
                        help="skip Holt-Winters; baselines only")
    parser.add_argument("--csv-out", help="write the per-series scores here")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if args.csv:
        frame = pd.read_csv(args.csv)
        missing = {args.series_col, args.date_col, args.value_col} - set(frame.columns)
        if missing:
            print(f"missing columns in {args.csv}: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
        series_col, date_col, value_col = args.series_col, args.date_col, args.value_col
        source = args.csv
    else:
        frame = _load_demo(args.level, args.country)
        series_col, date_col, value_col = "series", "period", "value"
        source = f"synthetic dataset, level={args.level}"

    series = bt.to_series(frame, series_col=series_col, date_col=date_col, value_col=value_col)

    config = bt.BacktestConfig(
        horizon=args.horizon,
        min_train=args.min_train,
        season=args.season,
        max_series=args.max_series,
    )

    models = default_models(season=args.season)
    if args.fast:
        models = [m for m in models if not isinstance(m, HoltWinters)]

    print(f"source: {source}")
    print(f"series: {len(series)}   horizon: {config.horizon} months   "
          f"min train: {config.min_train} months   season: {config.season}\n")

    result = bt.run(series, models=models, config=config)
    if not result.per_series:
        print("nothing scored — every series is too short for this configuration")
        for key, reason in list(result.skipped.items())[:5]:
            print(f"  {key}: {reason}")
        return 1

    summary = result.summary().sort_values("mase")
    print(summary.to_string(
        index=False,
        float_format=lambda v: f"{v:.2f}",
        columns=["model", "series", "mape", "wape", "mase", "bias", "beats_seasonal_naive"],
    ))

    scored = len({r.series for r in result.per_series})
    print(f"\n{scored} series scored, {len(result.skipped)} skipped as too short")
    print("mape/wape/bias in %, mase relative to seasonal naive (below 1.00 is better),")
    print("beats_seasonal_naive = share of series where this model's MASE wins")

    if args.csv_out:
        pd.DataFrame([
            {"series": r.series, "model": r.model, "origins": r.origins,
             "volume": r.volume, **r.scores.as_dict()}
            for r in result.per_series
        ]).to_csv(args.csv_out, index=False)
        print(f"\nper-series scores written to {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
