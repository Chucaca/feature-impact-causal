import numpy as np
import pandas as pd
import pytest

from feature_impact.consistency import monthly_consistency, trend_label
from feature_impact.psm import MIN_TREATED, estimate_att
from feature_impact.quadrants import CHAMPION, HIDDEN_GEM, LOW_LOW, WIDE_LOW, assign_quadrants, classify
from feature_impact.stratified import stratified_effect
from feature_impact.summary import STAGES, summarize, to_markdown
from feature_impact.synthetic import CAT_CONFOUNDERS, NUM_CONFOUNDERS, generate, treatment_col


@pytest.fixture(scope="module")
def small():
    return generate(n_per_month=6_000, n_months=1, seed=7)


def _truth(data, feature, outcome="renewed"):
    tr = data.truth
    return float(tr[(tr.feature == feature) & (tr.outcome == outcome)].true_att.iloc[0])


# ------------------------------------------------------------------------- synthetic
def test_generator_is_seeded_and_shaped():
    a = generate(n_per_month=500, n_months=2, seed=1)
    b = generate(n_per_month=500, n_months=2, seed=1)
    assert len(a.df) == 1000
    pd.testing.assert_frame_equal(a.df, b.df)
    null = a.truth[a.truth.true_logit_effect == 0]
    assert np.allclose(null.true_att, 0.0)


# ------------------------------------------------------------------------- PSM
@pytest.mark.parametrize("feature", ["auto_reorder", "e_invoice", "legacy_desktop_sync"])
def test_psm_recovers_known_effect(small, feature):
    res = estimate_att(small.df, treatment_col(feature), "renewed", CAT_CONFOUNDERS, NUM_CONFOUNDERS, n_boot=200)
    truth = _truth(small, feature)
    naive_err = abs(res.naive_diff - truth)
    assert res.status == "ok"
    assert abs(res.att - truth) < 0.04  # within tolerance on 6k rows
    assert abs(res.att - truth) < naive_err  # and much closer than the naive diff
    assert res.ci_low <= res.att <= res.ci_high


def test_psm_improves_balance(small):
    res = estimate_att(
        small.df, treatment_col("multi_branch_transfer"), "renewed", CAT_CONFOUNDERS, NUM_CONFOUNDERS, n_boot=50
    )
    bal = res.balance
    assert bal["smd_before"].abs().max() > 0.5  # strongly confounded feature
    assert bal["smd_after"].abs().mean() < bal["smd_before"].abs().mean() / 3
    assert res.n_imbalanced_after < int(bal["imbalanced_before"].sum())


def test_psm_without_replacement_uses_unique_controls(small):
    res = estimate_att(
        small.df, treatment_col("auto_reorder"), "renewed", CAT_CONFOUNDERS, NUM_CONFOUNDERS,
        replace=False, n_boot=50,
    )
    assert res.status == "ok"
    assert res.n_unique_controls == res.n_matched


def test_psm_guard_min_treated(small):
    df = small.df.copy()
    col = treatment_col("auto_reorder")
    df[col] = 0
    df.loc[df.index[: MIN_TREATED - 1], col] = 1
    res = estimate_att(df, col, "renewed", CAT_CONFOUNDERS, NUM_CONFOUNDERS)
    assert res.status == "insufficient_sample"
    assert np.isnan(res.att)


# ------------------------------------------------------------------------- stratified
def _hand_df():
    rows = []
    # stratum A: treated [1,1,0] mean 2/3 ; control [1,0,0] mean 1/3 -> effect 1/3, weight 3
    rows += [("A", 1, y) for y in (1, 1, 0)] + [("A", 0, y) for y in (1, 0, 0)]
    # stratum B: treated [1] mean 1 ; control [0,1,0,1] mean 1/2 -> effect 1/2, weight 1
    rows += [("B", 1, 1)] + [("B", 0, y) for y in (0, 1, 0, 1)]
    # stratum C: control all 0 (zero variance) -> skipped
    rows += [("C", 1, y) for y in (1, 1, 1)] + [("C", 0, y) for y in (0, 0, 0)]
    # stratum D: only 4 rows -> skipped as too small
    rows += [("D", 1, 1), ("D", 1, 1), ("D", 0, 0), ("D", 0, 1)]
    return pd.DataFrame(rows, columns=["s", "t", "y"])


def test_stratified_matches_hand_computation():
    res = stratified_effect(_hand_df(), "t", "y", ["s"], min_strata=1, min_coverage=0.0)
    expected = (3 * (1 / 3) + 1 * (1 / 2)) / 4  # = 0.375
    assert res.ate == pytest.approx(expected)
    assert res.n_strata_used == 2
    assert res.n_skipped_zero_var == 1
    assert res.n_skipped_small == 1
    assert res.treated_coverage == pytest.approx(4 / 9)
    # SE by hand: var_t/n_t + var_c/n_c (ddof=0) per stratum, weights 3 and 1
    v_a = (2 / 9) / 3 + (2 / 9) / 3
    v_b = 0 / 1 + 0.25 / 4
    assert res.se == pytest.approx(np.sqrt(9 * v_a + 1 * v_b) / 4)


def test_stratified_flags_not_interpretable():
    res = stratified_effect(_hand_df(), "t", "y", ["s"])  # defaults: >=3 strata, >=50% coverage
    assert not res.interpretable
    assert "usable strata" in res.note


# ------------------------------------------------------------------------- consistency
def test_consistency_rule_and_trend():
    monthly = pd.DataFrame(
        {
            "feature": ["a"] * 6 + ["b"] * 6,
            "snapshot": list(range(1, 7)) * 2,
            "est": [0.05, 0.05, 0.06, 0.05, 0.07, 0.09, 0.05, 0.0, 0.04, 0.0, 0.03, 0.0],
            "p": [0.01, 0.2, 0.01, 0.03, 0.001, 0.001, 0.01, 0.5, 0.02, 0.6, 0.04, 0.7],
        }
    )
    out = monthly_consistency(monthly, "est", "p").set_index("feature")
    assert out.loc["a", "months_significant"] == 5 and out.loc["a", "confirmed"]
    assert out.loc["b", "months_significant"] == 3 and not out.loc["b", "confirmed"]
    assert out.loc["a", "trend"] == "rising"  # 0.05 -> 0.07 -> 0.09
    assert trend_label(np.array([0.05, 0.051, 0.05]))[1] == "stable"
    assert trend_label(np.array([0.09, 0.06, 0.03]))[1] == "declining"


# ------------------------------------------------------------------------- quadrants
def test_quadrant_logic():
    assert classify(0.6, 0.05, True, adoption_threshold=0.3) == CHAMPION
    assert classify(0.1, 0.05, True, adoption_threshold=0.3) == HIDDEN_GEM
    assert classify(0.6, 0.05, False, adoption_threshold=0.3) == WIDE_LOW  # big but not confirmed
    assert classify(0.6, 0.0, True, adoption_threshold=0.3) == WIDE_LOW
    assert classify(0.1, -0.05, True, adoption_threshold=0.3) == LOW_LOW
    df = pd.DataFrame(
        {"adoption_rate": [0.1, 0.2, 0.5, 0.7], "psm_att": [0.05, -0.05, 0.0, 0.08], "confirmed": [True] * 4}
    )
    out = assign_quadrants(df)
    assert out["quadrant"].tolist() == [HIDDEN_GEM, LOW_LOW, WIDE_LOW, CHAMPION]
    assert out["harmful"].tolist() == [False, True, False, False]


# ------------------------------------------------------------------------- summary
def test_template_summary_offline(monkeypatch):
    monkeypatch.delenv("FEATURE_IMPACT_LLM", raising=False)
    table = pd.DataFrame(
        {
            "feature": ["f1", "f2", "f3"],
            "adoption_rate": [0.6, 0.1, 0.5],
            "psm_att": [0.05, 0.08, 0.0],
            "psm_ci_low": [0.03, 0.05, -0.02],
            "psm_ci_high": [0.07, 0.11, 0.02],
            "months_significant": [6, 5, 0],
            "confirmed": [True, True, False],
            "naive_confirmed": [True, True, True],
            "trend": ["stable", "rising", "n/a"],
            "aha_effect": [0.04, 0.06, 0.0],
            "aha_months_significant": [5, 6, 0],
            "aha_confirmed": [True, True, False],
        }
    )
    table = assign_quadrants(table, adoption_threshold=0.3)
    summary, source = summarize(table)
    assert source == "template"
    assert [s.stage for s in summary.stages] == list(STAGES)
    md = to_markdown(summary)
    assert "f2" in md and "Synthetic" in md


def test_unknown_provider_falls_back():
    table = pd.DataFrame(
        {
            "feature": ["f1"], "adoption_rate": [0.5], "psm_att": [0.0], "psm_ci_low": [0.0], "psm_ci_high": [0.0],
            "months_significant": [0], "confirmed": [False], "naive_confirmed": [False], "trend": ["n/a"],
            "aha_effect": [0.0], "aha_months_significant": [0], "aha_confirmed": [False],
        }
    )
    table = assign_quadrants(table)
    with pytest.warns(UserWarning):
        _, source = summarize(table, provider="no-such-provider")
    assert source == "template"
