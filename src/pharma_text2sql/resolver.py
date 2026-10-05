"""Entity resolution: map what the user said onto values the database holds.

A model asked for "sales of statins in the UK" will happily write
``WHERE country = 'UK' AND atc4 = 'statins'`` — and the query returns nothing,
which is the worst possible failure: it looks like an answer. Resolving the
terms *before* generating SQL turns a silent empty result into either the right
value or an explicit question back to the user.

Three outcomes per term, and the third is the one that matters:

* **resolved** — exactly one database value matches;
* **unresolved** — nothing matches, so the question is answered with "I don't
  know this molecule" instead of an empty table;
* **ambiguous** — several values match, so the user is asked which one.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

from . import schema

# Which column each entity type lives in.
ENTITY_COLUMNS = {
    "countries": "country",
    "regions": "region",
    "molecules": "molecule",
    "corporations": "corporation",
    "manufacturers": "manufacturer",
    "products": "product",
    "forms": "form",
    "channels": "channel",
    "atc1": "atc1",
    "atc2": "atc2",
    "atc3": "atc3",
    "atc4": "atc4",
}

# Common ways people name things that the database spells differently.
ALIASES = {
    "country": {
        "uk": "United Kingdom", "u.k.": "United Kingdom", "great britain": "United Kingdom",
        "britain": "United Kingdom", "england": "United Kingdom",
        "us": "United States", "u.s.": "United States", "usa": "United States",
        "états-unis": "United States", "etats-unis": "United States",
        "allemagne": "Germany", "espagne": "Spain", "italie": "Italy",
        "royaume-uni": "United Kingdom", "japon": "Japan", "maroc": "Morocco",
        "brésil": "Brazil", "bresil": "Brazil", "pologne": "Poland",
        "afrique du sud": "South Africa", "australie": "Australia",
    },
}

MIN_SIMILARITY = 0.78
MAX_SUGGESTIONS = 8
MIN_PROBE_LENGTH = 4


@dataclass
class Resolution:
    resolved: dict[str, list[str]] = field(default_factory=dict)
    unresolved: dict[str, list[str]] = field(default_factory=dict)
    ambiguous: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    @property
    def needs_clarification(self) -> bool:
        return bool(self.unresolved or self.ambiguous)

    def clarification(self) -> str:
        parts: list[str] = []
        for column, terms in self.unresolved.items():
            for term in terms:
                parts.append(f"I could not find “{term}” in {column}.")
        for column, per_term in self.ambiguous.items():
            for term, matches in per_term.items():
                options = ", ".join(matches[:MAX_SUGGESTIONS])
                parts.append(f"“{term}” matches several values in {column}: {options}. Which one?")
        return " ".join(parts)


# "the UK", "les statines" — an article in front of a value must not stop it
# from matching the value itself.
_ARTICLES = re.compile(r"^(the|le|la|les|l'|un|une|des|du|de la|de)\s+")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = re.sub(r"[\s\-_/]+", " ", text)
    return _ARTICLES.sub("", text).strip()


def resolve_term(term: str, column: str, values: list[str]) -> tuple[list[str], str]:
    """Resolve one term against one column.

    Returns ``(matches, status)`` with status in ``resolved`` / ``ambiguous`` /
    ``unresolved``.
    """
    needle = _normalise(term)
    if not needle:
        return [], "unresolved"

    alias = ALIASES.get(column, {}).get(needle)
    if alias and alias in values:
        return [alias], "resolved"

    index = {_normalise(v): v for v in values}

    # 1. exact match
    if needle in index:
        return [index[needle]], "resolved"

    # Below four characters a substring probe is pure noise: "of" sits inside
    # GLUCOFEN, "in" inside Spain. Short terms resolve by alias or exact match
    # only — which is what "UK" and "US" need and all they need.
    if len(needle) < MIN_PROBE_LENGTH:
        return [], "unresolved"

    # 2. substring — "statins" inside "C10AA — Statins", "sertralex" inside a
    #    product label. One hit resolves, several ask.
    contains = [orig for norm, orig in index.items() if needle in norm]
    if len(contains) == 1:
        return contains, "resolved"
    if len(contains) > 1:
        # An exact word match wins over a mere substring: "Tablet" should not be
        # ambiguous just because "Tablet 500 MG" also exists.
        exact_word = [v for v in contains if needle in _normalise(v).split()]
        if len(exact_word) == 1:
            return exact_word, "resolved"
        return sorted(contains)[:MAX_SUGGESTIONS], "ambiguous"

    # 3. fuzzy — typos, plurals, accents already stripped
    close = difflib.get_close_matches(needle, list(index), n=MAX_SUGGESTIONS, cutoff=MIN_SIMILARITY)
    if len(close) == 1:
        return [index[close[0]]], "resolved"
    if len(close) > 1:
        return [index[c] for c in close], "ambiguous"

    return [], "unresolved"


def resolve(entities: dict[str, list[str]]) -> Resolution:
    """Resolve every extracted entity. ``entities`` is ``{entity_type: [terms]}``."""
    out = Resolution()
    values_by_column = schema.get().values

    for entity_type, terms in (entities or {}).items():
        # A term extracted without a type: try every indexed column before
        # declaring it unknown.
        if entity_type == "unknown":
            for term in terms:
                _resolve_untyped(term, values_by_column, out)
            continue

        column = ENTITY_COLUMNS.get(entity_type, entity_type)
        values = values_by_column.get(column)
        if not values:
            continue

        for term in terms:
            matches, status = resolve_term(term, column, values)
            if status == "resolved":
                out.resolved.setdefault(column, []).extend(matches)
            elif status == "ambiguous":
                out.ambiguous.setdefault(column, {})[term] = matches
            else:
                out.unresolved.setdefault(column, []).append(term)

    return out


def _resolve_untyped(term: str, values_by_column: dict[str, list[str]], out: Resolution) -> None:
    """Resolve a term whose entity type is unknown, across every column.

    One column matching settles it. Several columns matching is ambiguity worth
    surfacing, not a coin toss. No column matching is the important outcome: the
    term is reported so the user is asked about it rather than ignored.
    """
    hits: dict[str, list[str]] = {}
    for column, values in values_by_column.items():
        if column not in ENTITY_COLUMNS.values():
            continue
        matches, status = resolve_term(term, column, values)
        if status == "resolved":
            hits[column] = matches
        elif status == "ambiguous":
            out.ambiguous.setdefault(column, {})[term] = matches
            return

    if len(hits) == 1:
        column, matches = next(iter(hits.items()))
        out.resolved.setdefault(column, []).extend(matches)
    elif len(hits) > 1:
        out.ambiguous.setdefault("several columns", {})[term] = [
            f"{column}={matches[0]}" for column, matches in hits.items()
        ]
    else:
        out.unresolved.setdefault("the data", []).append(term)
