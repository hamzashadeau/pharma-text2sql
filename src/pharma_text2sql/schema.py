"""Schema and value index.

Two things the model needs and the database can give cheaply:

* the **column list with types**, injected into the prompt so the model writes
  SQL against real columns rather than invented ones;
* a **value index** — the distinct values of every categorical column — which
  lets the resolver map "statins" or "the UK" onto a value that actually exists
  before any SQL is written.

Both are built once and cached on disk. Rebuilding them against a large table
costs minutes; reading the cache costs milliseconds, which is the difference
between a service that restarts instantly and one that does not.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb

from .config import settings

logger = logging.getLogger(__name__)

CACHE_PATH = settings.dataset_path.with_suffix(".schema.json")

# Columns we index values for. Numeric and high-cardinality columns are left
# out: nobody asks a question by naming a row's unit count.
CATEGORICAL = (
    "country", "region", "sub_region",
    "atc1", "atc2", "atc3", "atc4",
    "molecule", "corporation", "manufacturer", "product",
    "form", "strength", "channel", "period", "quarter",
)

# Above this many distinct values a column is described but not indexed, to keep
# the cache small. The resolver then falls back to a LIKE probe.
MAX_INDEXED_VALUES = 5_000


@dataclass(frozen=True)
class Column:
    name: str
    type: str


@dataclass
class Schema:
    columns: list[Column]
    values: dict[str, list[str]]
    row_count: int

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def describe(self) -> str:
        """Compact description injected into the model prompt."""
        lines = [f"Table {settings.table_name} ({self.row_count:,} rows)", "Columns:"]
        for col in self.columns:
            sample = self.values.get(col.name, [])[:4]
            hint = f"  e.g. {', '.join(sample)}" if sample else ""
            lines.append(f"  - {col.name} ({col.type}){hint}")
        return "\n".join(lines)


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    # CREATE VIEW cannot be prepared in DuckDB, so the path is inlined — quoted
    # the SQL way, with embedded apostrophes doubled.
    path = str(settings.dataset_path).replace("'", "''")
    con.execute(f"CREATE VIEW {settings.table_name} AS SELECT * FROM read_parquet('{path}')")
    return con


def build(force: bool = False) -> Schema:
    """Return the schema, from cache when available."""
    if CACHE_PATH.exists() and not force:
        try:
            return _load(CACHE_PATH)
        except Exception:  # a corrupt cache must not take the service down
            logger.warning("schema cache unreadable, rebuilding", exc_info=True)

    if not settings.dataset_path.exists():
        raise FileNotFoundError(
            f"dataset not found: {settings.dataset_path}\nRun: python data/generate.py"
        )

    con = _connect()
    try:
        described = con.execute(f"DESCRIBE {settings.table_name}").fetchall()
        columns = [Column(name=r[0], type=r[1]) for r in described]
        row_count = con.execute(f"SELECT count(*) FROM {settings.table_name}").fetchone()[0]

        values: dict[str, list[str]] = {}
        names = {c.name for c in columns}
        for name in CATEGORICAL:
            if name not in names:
                continue
            distinct = con.execute(
                f'SELECT DISTINCT "{name}" FROM {settings.table_name} '
                f'WHERE "{name}" IS NOT NULL ORDER BY 1 LIMIT {MAX_INDEXED_VALUES + 1}'
            ).fetchall()
            if len(distinct) > MAX_INDEXED_VALUES:
                logger.info("column %s has too many values to index", name)
                continue
            values[name] = [str(row[0]) for row in distinct]
    finally:
        con.close()

    schema = Schema(columns=columns, values=values, row_count=row_count)
    _save(schema, CACHE_PATH)
    return schema


def _save(schema: Schema, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "columns": [{"name": c.name, "type": c.type} for c in schema.columns],
                "values": schema.values,
                "row_count": schema.row_count,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _load(path: Path) -> Schema:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Schema(
        columns=[Column(**c) for c in raw["columns"]],
        values=raw["values"],
        row_count=raw["row_count"],
    )


_schema: Schema | None = None


def get() -> Schema:
    """Process-wide cached schema."""
    global _schema
    if _schema is None:
        _schema = build()
    return _schema


def query(sql: str) -> "object":
    """Run a validated query and return a DataFrame.

    Everything reaching this function has been through :mod:`sql_guard`.
    """
    con = _connect()
    try:
        return con.execute(sql).df()
    finally:
        con.close()
