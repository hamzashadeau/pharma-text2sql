"""Step 1 — turn a question into a structured intent.

The model is asked for JSON, never for prose: a shape that can be validated is
a shape the rest of the pipeline can rely on. When no API key is configured the
same structure is produced by a lexical extractor, so every downstream stage —
resolution, generation, guard, execution — is exercised identically offline.

What is extracted is deliberately *not* SQL. Entities come out as the user
spelled them ("statins", "the UK", "sertralx") and are resolved against real
database values in the next step. Letting the model jump straight to SQL is
what produces a syntactically perfect query returning zero rows.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from . import llm, schema
from .resolver import ENTITY_COLUMNS, _normalise, resolve_term

logger = logging.getLogger(__name__)

# Metric wording -> (column, aggregate)
METRICS = {
    "sales": ("sales_eur", "sum"), "revenue": ("sales_eur", "sum"),
    "turnover": ("sales_eur", "sum"), "value": ("sales_eur", "sum"),
    "chiffre d'affaires": ("sales_eur", "sum"), "ventes": ("sales_eur", "sum"),
    "units": ("units", "sum"), "volume": ("units", "sum"),
    "unités": ("units", "sum"), "quantity": ("units", "sum"),
}

# Grouping wording -> column
DIMENSIONS = {
    "country": "country", "countries": "country", "pays": "country",
    "region": "region", "regions": "region",
    "sub region": "sub_region", "sub-region": "sub_region",
    "molecule": "molecule", "molecules": "molecule", "molécule": "molecule",
    "corporation": "corporation", "corporations": "corporation", "company": "corporation",
    "companies": "corporation", "laboratoire": "corporation",
    "manufacturer": "manufacturer", "manufacturers": "manufacturer",
    "product": "product", "products": "product", "presentation": "product",
    "presentations": "product", "produit": "product",
    "form": "form", "forms": "form", "forme": "form",
    "strength": "strength", "dosage": "strength",
    "channel": "channel", "channels": "channel", "canal": "channel",
    "atc1": "atc1", "atc2": "atc2", "atc3": "atc3", "atc4": "atc4",
    "class": "atc4", "classes": "atc4",
    "year": "year", "years": "year", "année": "year", "annee": "year",
    "quarter": "quarter", "quarters": "quarter", "trimestre": "quarter",
    "month": "period", "months": "period", "mois": "period",
    "period": "period",
}

# "monthly sales" names its grouping without a "by".
GRANULARITY = (
    (r"\b(monthly|per month|par mois|mensuel\w*)\b", "period"),
    (r"\b(quarterly|per quarter|par trimestre|trimestriel\w*)\b", "quarter"),
    (r"\b(yearly|annually|annual|per year|par an|annuel\w*)\b", "year"),
    (r"\b(over time|month by month|au fil du temps)\b", "period"),
)

ADVANCED_MARKERS = (
    "growth", "croissance", "year over year", "yoy", "evolution", "évolution",
    "share", "part de marché", "market share", "trend", "tendance",
    "correlation", "forecast", "prévision", "cagr",
)


@dataclass
class Extraction:
    question: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    metric: str = "sales_eur"
    aggregate: str = "sum"
    group_by: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    top_n: int | None = None
    ascending: bool = False
    query_type: str = "aggregation"  # aggregation | ranking | trend | advanced

    @property
    def is_advanced(self) -> bool:
        return self.query_type == "advanced"


SYSTEM_PROMPT = """You convert pharmaceutical market questions into a JSON intent.
Return ONLY a JSON object with these keys:
  entities    object mapping entity type to the terms the user used, verbatim.
              Entity types: countries, regions, molecules, corporations,
              manufacturers, products, forms, channels, atc1, atc2, atc3, atc4.
  metric      one of: sales_eur, units
  aggregate   one of: sum, avg, count
  group_by    list of column names to group by (country, molecule, corporation,
              manufacturer, product, form, strength, channel, atc1..atc4,
              year, quarter, period)
  years       list of integer years mentioned, [] if none
  top_n       integer if the user asked for a top/bottom N, else null
  ascending   true if the user asked for the lowest/bottom, else false
  query_type  one of: aggregation, ranking, trend, advanced

Do NOT write SQL. Do NOT correct the spelling of entity terms — copy them as
written; they are resolved against the database separately."""


def extract(question: str) -> Extraction:
    """Extract intent, via the model when one is configured."""
    if llm.available():
        try:
            return _extract_llm(question)
        except Exception:
            logger.warning("model extraction failed, using lexical extractor", exc_info=True)
    return _extract_lexical(question)


def _extract_llm(question: str) -> Extraction:
    raw = llm.complete_json(SYSTEM_PROMPT, question)
    return Extraction(
        question=question,
        entities={k: list(v) for k, v in (raw.get("entities") or {}).items() if v},
        metric=raw.get("metric") or "sales_eur",
        aggregate=raw.get("aggregate") or "sum",
        group_by=list(raw.get("group_by") or []),
        years=[int(y) for y in (raw.get("years") or [])],
        top_n=raw.get("top_n"),
        ascending=bool(raw.get("ascending")),
        query_type=raw.get("query_type") or "aggregation",
    )


# ---------------------------------------------------------------------------
# Lexical extractor — the offline path
# ---------------------------------------------------------------------------

def _extract_lexical(question: str) -> Extraction:
    q = _normalise(question)
    out = Extraction(question=question)

    for wording, (column, aggregate) in METRICS.items():
        if wording in q:
            out.metric, out.aggregate = column, aggregate
            break

    if re.search(r"\b(how many|number of|count|combien)\b", q):
        out.aggregate = "count"
    elif re.search(r"\b(average|avg|mean|moyenne)\b", q):
        out.aggregate = "avg"

    out.years = sorted({int(y) for y in re.findall(r"\b(20\d{2})\b", question)})

    top = re.search(r"\b(?:top|best|highest|bottom|lowest|worst)\s+(\d+)\b", q)
    if top:
        out.top_n = int(top.group(1))
    elif re.search(r"\b(top|best|highest|bottom|lowest|worst|which .* most|quel .* plus)\b", q):
        out.top_n = 10
    out.ascending = bool(re.search(r"\b(bottom|lowest|worst|least|moins)\b", q))

    for pattern, column in GRANULARITY:
        if re.search(pattern, q) and column not in out.group_by:
            out.group_by.append(column)

    for wording, column in _dimension_patterns():
        if re.search(rf"\b(?:by|per|for each|par|pour chaque)\s+{re.escape(wording)}\b", q):
            if column not in out.group_by:
                out.group_by.append(column)

    if not out.group_by and out.top_n:
        # "top 10 molecules" — the dimension is named without "by".
        for wording, column in _dimension_patterns():
            if re.search(rf"\b{re.escape(wording)}\b", q):
                out.group_by.append(column)
                break

    if any(marker in q for marker in ADVANCED_MARKERS):
        out.query_type = "advanced"
    elif out.top_n:
        out.query_type = "ranking"
    elif any(c in out.group_by for c in ("year", "quarter", "period")):
        out.query_type = "trend"

    out.entities = _candidate_entities(question)
    return out


def _dimension_patterns() -> list[tuple[str, str]]:
    # Longest wording first, so "sub region" is not eaten by "region".
    return sorted(DIMENSIONS.items(), key=lambda kv: -len(kv[0]))


def _candidate_entities(question: str) -> dict[str, list[str]]:
    """Pull candidate entity terms out of the raw question.

    Word windows of three, two then one word are probed against the value
    index, longest first; once a window matches, the words it covers are
    claimed so a shorter window inside it cannot match the same thing twice.
    Terms come out as the user wrote them — resolution is a separate step.
    """
    words = re.findall(r"[\w'\u2019\-\.]+", question)
    values_by_column = schema.get().values
    column_to_type = {v: k for k, v in ENTITY_COLUMNS.items()}

    found: dict[str, list[str]] = {}
    claimed: set[int] = set()

    for size in (3, 2, 1):
        for start in range(len(words) - size + 1):
            span = range(start, start + size)
            if claimed.intersection(span):
                continue

            window = " ".join(words[start:start + size])
            norm = _normalise(window)
            if len(norm) < 2 or norm in _STOPWORDS:
                continue

            for column, values in values_by_column.items():
                entity_type = column_to_type.get(column)
                if entity_type is None:
                    continue

                matches, status = resolve_term(window, column, values)
                if status in ("resolved", "ambiguous") and matches:
                    found.setdefault(entity_type, []).append(window)
                    claimed.update(span)
                    break

    # A capitalised word that matched nothing is the case that must NOT be
    # dropped silently: "sales of NOTAMOLECULE in Spain" would otherwise become
    # a confident total over every molecule in Spain. It is handed to the
    # resolver without a type — the resolver searches every column and, finding
    # nothing anywhere, turns it into a question back to the user.
    for index, word in enumerate(words):
        if index == 0 or index in claimed:
            continue
        if not re.fullmatch(r"[A-Z][\w\-]{3,}", word):
            continue
        norm = _normalise(word)
        if norm in _STOPWORDS or norm in DIMENSIONS or norm in METRICS:
            continue
        found.setdefault("unknown", []).append(word)

    return found


_STOPWORDS = {
    "the", "and", "for", "with", "per", "all", "top", "best", "most", "sales",
    "units", "volume", "total", "show", "give", "list", "what", "which", "how",
    "many", "much", "was", "were", "are", "its", "our", "their", "les", "des",
    "pour", "avec", "quel", "quelle", "quels", "combien", "ventes", "total",
    "par", "dans", "sur", "plus", "moins", "annee", "year", "years",
}
