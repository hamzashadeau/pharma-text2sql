"""Comparing two result sets — the core of execution accuracy.

Two queries answer the same question when they return the same data. Comparing
the SQL text instead is meaningless: a dozen correct formulations exist for any
aggregate, and an LLM picks a different one every run.

Four things have to be forgiven, because none of them changes the answer:

* **row order** — without an ``ORDER BY`` the engine may return rows in any
  order, and even with one, ties break arbitrarily;
* **column names** — ``total_sales_eur`` and ``TotalValueEUR`` are the same
  column with a different alias, and the alias is the model's free choice;
* **column order** — ``SELECT country, total`` and ``SELECT total, country``
  are the same result presented differently (switch off with
  ``ignore_column_order=False`` when the order carries meaning);
* **floating-point noise** — sums over millions of rows differ in the last
  digits depending on the order the engine added them up.

Nothing else is forgiven. A different row count, a missing group, a figure off
by a percent: those are different answers.

A fingerprint is provided for the case that matters operationally — comparing
against a reference computed last week, without re-running a query over
hundreds of millions of rows, and without storing the data itself.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

# Sums over large tables differ in the last digits depending on the order the
# engine added them. 0.5% is far below any difference a reader would notice and
# far above that noise.
DEFAULT_TOLERANCE = 0.005
ROUNDING = 6


def normalise(
    frame: pd.DataFrame,
    *,
    ignore_column_order: bool = True,
) -> list[tuple]:
    """Canonical, comparable view of a result set."""
    if frame is None or len(frame) == 0:
        return []

    columns: list[list] = []
    for column in frame.columns:
        values = []
        for cell in frame[column]:
            if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                values.append(round(float(cell), ROUNDING))
            elif cell is None or (isinstance(cell, float) and pd.isna(cell)):
                values.append(None)
            else:
                values.append(str(cell).strip())
        columns.append(values)

    if ignore_column_order:
        # Sorted by the column's own content, so the same data lines up
        # whatever order the SELECT listed it in.
        columns.sort(key=lambda values: json.dumps(values, default=str, sort_keys=True))

    rows = [tuple(column[i] for column in columns) for i in range(len(frame))]
    return sorted(rows, key=lambda row: json.dumps(row, default=str))


def equivalent(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    ignore_column_order: bool = True,
) -> bool:
    """True when two result sets carry the same answer."""
    a = normalise(left, ignore_column_order=ignore_column_order)
    b = normalise(right, ignore_column_order=ignore_column_order)

    if len(a) != len(b):
        return False

    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            return False
        for cell_a, cell_b in zip(row_a, row_b):
            if isinstance(cell_a, float) and isinstance(cell_b, float):
                scale = max(abs(cell_a), abs(cell_b), 1.0)
                if abs(cell_a - cell_b) / scale > tolerance:
                    return False
            elif cell_a != cell_b:
                return False
    return True


def fingerprint(frame: pd.DataFrame, *, ignore_column_order: bool = True) -> str:
    """Short digest of a result set.

    Lets a run be compared against a reference taken last week without
    re-running a query over hundreds of millions of rows — and without keeping
    the data, which on a real corpus is the difference between a reference file
    that can be shared and one that cannot.

    Floating-point values are rounded to six places before hashing, so the
    digest is stable across engines; two results within the tolerance but on
    opposite sides of a rounding boundary will still differ, which is why
    :func:`equivalent` exists and the digest is only a fast path.
    """
    payload = json.dumps(
        normalise(frame, ignore_column_order=ignore_column_order),
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def describe(frame: pd.DataFrame) -> str:
    if frame is None:
        return "no result"
    if len(frame) == 0:
        return "0 rows"
    return f"{len(frame)} rows x {len(frame.columns)} columns"
