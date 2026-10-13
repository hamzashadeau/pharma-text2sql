"""Result-set comparison: what must be forgiven, and what must not."""

from __future__ import annotations

import pandas as pd

from pharma_text2sql.evaluation import resultset as rs

BASE = pd.DataFrame({"country": ["France", "Spain", "Japan"], "total": [10.0, 20.0, 30.0]})


def test_identical_results_match():
    assert rs.equivalent(BASE, BASE.copy())


def test_row_order_is_forgiven():
    """Without an ORDER BY the engine may return rows in any order, and even
    with one, ties break arbitrarily."""
    shuffled = BASE.iloc[[2, 0, 1]].reset_index(drop=True)
    assert rs.equivalent(BASE, shuffled)


def test_column_names_are_forgiven():
    """The alias is the model's free choice; it carries no answer."""
    renamed = BASE.rename(columns={"country": "Country", "total": "TotalValueEUR"})
    assert rs.equivalent(BASE, renamed)


def test_column_order_is_forgiven_by_default():
    swapped = BASE[["total", "country"]]
    assert rs.equivalent(BASE, swapped)


def test_column_order_can_be_enforced():
    swapped = BASE[["total", "country"]]
    assert not rs.equivalent(BASE, swapped, ignore_column_order=False)


def test_floating_point_noise_is_forgiven():
    """Sums over millions of rows differ in the last digits with the order the
    engine added them up."""
    noisy = BASE.copy()
    noisy["total"] = noisy["total"] * (1 + 1e-9)
    assert rs.equivalent(BASE, noisy)


def test_a_real_difference_is_not_forgiven():
    changed = BASE.copy()
    changed.loc[0, "total"] = 10.5          # 5% off
    assert not rs.equivalent(BASE, changed)


def test_a_missing_row_is_not_forgiven():
    assert not rs.equivalent(BASE, BASE.iloc[:2])


def test_an_extra_column_is_not_forgiven():
    extra = BASE.copy()
    extra["units"] = [1, 2, 3]
    assert not rs.equivalent(BASE, extra)


def test_empty_results_match_each_other():
    assert rs.equivalent(pd.DataFrame(), pd.DataFrame())


def test_empty_does_not_match_populated():
    """The failure mode this whole project keeps circling back to: an empty
    result looks like an answer."""
    assert not rs.equivalent(BASE, pd.DataFrame())


def test_fingerprints_agree_with_equivalence():
    shuffled = BASE.iloc[[1, 2, 0]].reset_index(drop=True)
    renamed = shuffled.rename(columns={"total": "SomethingElse"})

    assert rs.fingerprint(BASE) == rs.fingerprint(renamed)

    changed = BASE.copy()
    changed.loc[1, "total"] = 999.0
    assert rs.fingerprint(BASE) != rs.fingerprint(changed)


def test_fingerprint_is_stable_across_calls():
    assert rs.fingerprint(BASE) == rs.fingerprint(BASE.copy())


def test_strings_and_numbers_do_not_collide():
    numeric = pd.DataFrame({"x": [1.0, 2.0]})
    textual = pd.DataFrame({"x": ["1.0", "2.0"]})
    assert not rs.equivalent(numeric, textual)
