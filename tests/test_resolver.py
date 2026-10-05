"""Resolution decides whether an answer is right, wrong, or an honest question."""

from __future__ import annotations

from pharma_text2sql import resolver

COUNTRIES = ["France", "Germany", "Spain", "United Kingdom", "United States", "Japan"]
MOLECULES = ["ATORVASTIN", "ROSUVALEN", "SIMVACOR", "SERTRALEX", "GLUCOFEN"]
ATC4 = ["C10AA — Statins", "N06AB — SSRIs", "A10BA — Biguanides"]


def test_exact_match():
    matches, status = resolver.resolve_term("France", "country", COUNTRIES)
    assert status == "resolved"
    assert matches == ["France"]


def test_alias():
    matches, status = resolver.resolve_term("UK", "country", COUNTRIES)
    assert status == "resolved"
    assert matches == ["United Kingdom"]


def test_article_is_ignored():
    matches, status = resolver.resolve_term("the UK", "country", COUNTRIES)
    assert status == "resolved"
    assert matches == ["United Kingdom"]


def test_case_and_accents_are_ignored():
    _, status = resolver.resolve_term("fRaNcE", "country", COUNTRIES)
    assert status == "resolved"


def test_substring_inside_a_label():
    matches, status = resolver.resolve_term("statins", "atc4", ATC4)
    assert status == "resolved"
    assert matches == ["C10AA — Statins"]


def test_typo_is_recovered():
    matches, status = resolver.resolve_term("ATORVASTINE", "molecule", MOLECULES)
    assert status == "resolved"
    assert matches == ["ATORVASTIN"]


def test_ambiguity_is_surfaced_not_guessed():
    matches, status = resolver.resolve_term("United", "country", COUNTRIES)
    assert status == "ambiguous"
    assert set(matches) == {"United Kingdom", "United States"}


def test_unknown_term_is_reported():
    matches, status = resolver.resolve_term("Atlantis", "country", COUNTRIES)
    assert status == "unresolved"
    assert matches == []


def test_short_terms_do_not_match_by_substring():
    """"of" sits inside GLUCOFEN, "in" inside Spain.

    Before the minimum probe length, "sales of X" silently resolved "of" to a
    molecule and filtered the whole query on it — a wrong answer that looked
    perfectly normal.
    """
    _, status = resolver.resolve_term("of", "molecule", MOLECULES)
    assert status == "unresolved"

    _, status = resolver.resolve_term("in", "country", COUNTRIES)
    assert status == "unresolved"


def test_resolution_groups_outcomes_by_column():
    result = resolver.resolve({"countries": ["France"], "molecules": ["NOPE"]})
    assert result.resolved.get("country") == ["France"]
    assert "NOPE" in result.unresolved.get("molecule", [])
    assert result.needs_clarification
    assert "NOPE" in result.clarification()
