"""Does the narration only cite figures the result table actually contains?

This is the failure that matters most in an analytics assistant and the one a
status counter cannot see. The SQL runs, the table is right, and the sentence
underneath says "sales grew to 4.2 million" when no cell holds 4.2 million. The
reader quotes the sentence, not the table.

The check is deliberately **deterministic**: every number in the narration is
extracted and matched against the result. No model is asked whether the model
was honest — that is circular, it costs money per question, and it cannot run
in CI on every prompt change. A judge still has a place for things arithmetic
cannot see (is this the right answer to the question?), and that lives in
:mod:`judge`. This module answers the narrower question completely.

What counts as supported:

* any cell of the result table, within a relative tolerance (narration rounds:
  "37.7 million" for 37 680 097.44 is correct, not a hallucination);
* the number of rows ("5 countries");
* the total of any numeric column, since narration routinely sums a column the
  table shows broken down.

What is excluded rather than failed:

* years and other values already present in the question — they are context,
  not claims about the data;
* percentages, which are usually derived (shares, growth) and would need the
  derivation to verify. They are reported separately as unchecked, never
  silently counted as supported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

# 1 234 567,89 / 1,234,567.89 / 37.7 / 4.2M / 3 billion
_NUMBER = re.compile(
    r"""
    (?P<number>
        \d{1,3}(?:[  ,  ]\d{3})+(?:[.,]\d+)?   # grouped thousands
      | \d+(?:[.,]\d+)?                                    # plain
    )
    \s*
    (?P<suffix>%|percent|pour\s?cent|k\b|m\b|million[s]?|bn\b|billion[s]?|md\b|milliard[s]?)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SCALES = {
    "k": 1e3,
    "m": 1e6, "million": 1e6, "millions": 1e6,
    "bn": 1e9, "billion": 1e9, "billions": 1e9,
    "md": 1e9, "milliard": 1e9, "milliards": 1e9,
}

# Narration rounds. 37.7 million for 37 680 097.44 is a correct statement.
DEFAULT_TOLERANCE = 0.01
YEAR_RANGE = range(1900, 2101)


@dataclass
class Citation:
    text: str
    value: float
    is_percentage: bool = False


@dataclass
class Faithfulness:
    supported: list[Citation] = field(default_factory=list)
    unsupported: list[Citation] = field(default_factory=list)
    unchecked: list[Citation] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return len(self.supported) + len(self.unsupported)

    @property
    def ok(self) -> bool:
        """True when nothing citable was cited wrongly.

        A narration with no figures at all is faithful — it claims nothing.
        """
        return not self.unsupported

    @property
    def score(self) -> float:
        """Share of checkable figures that the table supports. 1.0 if none."""
        return 1.0 if self.checked == 0 else len(self.supported) / self.checked

    def summary(self) -> str:
        if not self.checked and not self.unchecked:
            return "no figures cited"
        parts = [f"{len(self.supported)}/{self.checked} figures supported"]
        if self.unsupported:
            parts.append("unsupported: " + ", ".join(c.text for c in self.unsupported[:5]))
        if self.unchecked:
            parts.append(f"{len(self.unchecked)} unchecked (derived)")
        return "; ".join(parts)


def _to_float(raw: str) -> float | None:
    text = raw.strip().replace(" ", " ").replace(" ", " ")

    # Decide which separator is the decimal one. "1,234" is ambiguous on its
    # own; the shape of the groups settles it.
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):      # 1.234.567,89
            text = text.replace(".", "").replace(" ", "").replace(",", ".")
        else:                                       # 1,234,567.89
            text = text.replace(",", "").replace(" ", "")
    elif "," in text:
        before, _, after = text.rpartition(",")
        if len(after) == 3 and before.replace(" ", "").isdigit():
            text = text.replace(",", "").replace(" ", "")   # 1,234 → thousands
        else:
            text = text.replace(" ", "").replace(",", ".")  # 12,5 → decimal
    else:
        text = text.replace(" ", "")

    try:
        return float(text)
    except ValueError:
        return None


def citations(narration: str) -> list[Citation]:
    """Every number the narration states, with its scale applied."""
    out: list[Citation] = []
    for match in _NUMBER.finditer(narration or ""):
        value = _to_float(match.group("number"))
        if value is None:
            continue

        suffix = (match.group("suffix") or "").strip().lower()
        is_percentage = suffix in {"%", "percent", "pour cent", "pourcent"}
        if not is_percentage and suffix:
            value *= _SCALES.get(suffix.rstrip("."), 1.0)

        out.append(Citation(text=match.group(0).strip(), value=value, is_percentage=is_percentage))
    return out


def _mask_labels(narration: str, frame: pd.DataFrame) -> str:
    """Blank out label text the table contains, before reading figures.

    "Leading product: Aciclovex 20 MG" and "sales peaked in 2024-01" both carry
    digits that are part of a *name* the result already holds, not claims about
    a quantity. Flagging them as unsupported figures would fill the report with
    noise, and a check that cries wolf is a check people stop reading.

    Longest labels first, so masking "2024-01" is not pre-empted by a shorter
    label that overlaps it.
    """
    labels: set[str] = set()
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        labels.update(str(v) for v in series.dropna().unique() if any(c.isdigit() for c in str(v)))

    for label in sorted(labels, key=len, reverse=True):
        narration = narration.replace(label, " ")
    return narration


def _supported_values(frame: pd.DataFrame) -> list[float]:
    values: list[float] = [float(len(frame))]
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        values.extend(float(v) for v in series)
        # Narration routinely totals a column the table shows broken down.
        values.append(float(series.sum()))
    return values


def check(
    narration: str,
    frame: pd.DataFrame,
    *,
    question: str = "",
    tolerance: float = DEFAULT_TOLERANCE,
) -> Faithfulness:
    """Check every figure in ``narration`` against ``frame``."""
    result = Faithfulness()
    if frame is None:
        frame = pd.DataFrame()

    available = _supported_values(frame)
    question_numbers = {c.value for c in citations(question)}
    narration = _mask_labels(narration or "", frame)

    for citation in citations(narration):
        if citation.is_percentage:
            # Derived unless it happens to be in the table; never assumed right.
            if _matches(citation.value, available, tolerance):
                result.supported.append(citation)
            else:
                result.unchecked.append(citation)
            continue

        # Years and figures restated from the question are context, not claims.
        if citation.value in question_numbers:
            continue
        if float(citation.value).is_integer() and int(citation.value) in YEAR_RANGE:
            if not _matches(citation.value, available, tolerance):
                continue

        if _matches(citation.value, available, tolerance):
            result.supported.append(citation)
        else:
            result.unsupported.append(citation)

    return result


def _matches(value: float, available: list[float], tolerance: float) -> bool:
    for candidate in available:
        scale = max(abs(value), abs(candidate), 1.0)
        if abs(value - candidate) / scale <= tolerance:
            return True
    return False
