"""End-to-end monthly pipeline + evaluation against the synthetic ground truth."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .consistency import monthly_consistency
from .psm import estimate_att, naive_diff
from .quadrants import assign_quadrants
from .stratified import stratified_effect
from .synthetic import CAT_CONFOUNDERS, NUM_CONFOUNDERS, OUTCOMES, STRATA_COLS, treatment_col


def run_monthly(
    df: pd.DataFrame,
    features: list[str],
    outcomes: tuple[str, ...] = OUTCOMES,
    truth: pd.DataFrame | None = None,
    n_boot: int = 500,
    seed: int = 0,
    progress: bool = False,
) -> pd.DataFrame:
    """Run naive, PSM and stratified estimators per snapshot x feature x outcome (long format)."""
    rows = []
    for snap, m in df.groupby("snapshot", sort=True):
        if progress:
            print(f"  snapshot {snap}: {len(m):,} merchant-months")
        for feat in features:
            tcol = treatment_col(feat)
            for outcome in outcomes:
                nv = naive_diff(m, tcol, outcome)
                ps = estimate_att(m, tcol, outcome, CAT_CONFOUNDERS, NUM_CONFOUNDERS, n_boot=n_boot, seed=seed)
                st = stratified_effect(m, tcol, outcome, STRATA_COLS)
                st_all = stratified_effect(m, tcol, outcome, STRATA_COLS, skip_zero_var=False)
                rows.append(
                    {
                        "snapshot": snap,
                        "feature": feat,
                        "outcome": outcome,
                        "adoption_rate": float(m[tcol].mean()),
                        **nv,
                        "psm_status": ps.status,
                        "psm_att": ps.att,
                        "psm_lift": ps.lift,
                        "psm_se": ps.se,
                        "psm_ci_low": ps.ci_low,
                        "psm_ci_high": ps.ci_high,
                        "psm_p": ps.p_value,
                        "n_treated": ps.n_treated,
                        "n_matched": ps.n_matched,
                        "n_unmatched": ps.n_unmatched,
                        "max_abs_smd_before": ps.max_abs_smd_before,
                        "max_abs_smd_after": ps.max_abs_smd_after,
                        "n_imbalanced_after": ps.n_imbalanced_after,
                        "strat_ate": st.ate,
                        "strat_se": st.se,
                        "strat_ci_low": st.ci_low,
                        "strat_ci_high": st.ci_high,
                        "strat_p": st.p_value,
                        "strat_strata_used": st.n_strata_used,
                        "strat_treated_coverage": st.treated_coverage,
                        "strat_interpretable": st.interpretable,
                        "strat_keepzv_ate": st_all.ate,
                        "strat_keepzv_ci_low": st_all.ci_low,
                        "strat_keepzv_ci_high": st_all.ci_high,
                        "strat_keepzv_p": st_all.p_value,
                    }
                )
    out = pd.DataFrame(rows)
    if truth is not None:
        out = out.merge(
            truth[["snapshot", "feature", "outcome", "true_att", "true_logit_effect"]],
            on=["snapshot", "feature", "outcome"],
            how="left",
        )
    return out


def _pooled_ci(g: pd.DataFrame, est: str, se: str) -> tuple[float, float, float]:
    """Mean of independent monthly estimates and its 95% CI (cohorts are disjoint)."""
    e, s = g[est].to_numpy(float), g[se].to_numpy(float)
    ok = ~np.isnan(e) & ~np.isnan(s)
    if not ok.any():
        return np.nan, np.nan, np.nan
    mean = e[ok].mean()
    half = 1.96 * np.sqrt(np.sum(s[ok] ** 2)) / ok.sum()
    return float(mean), float(mean - half), float(mean + half)


def feature_table(monthly: pd.DataFrame, min_sig_months: int = 4) -> pd.DataFrame:
    """One row per feature: pooled estimates, consistency labels and quadrant."""
    ren = monthly[monthly["outcome"] == "renewed"]
    aha = monthly[monthly["outcome"] == "aha"]

    psm_c = monthly_consistency(ren, "psm_att", "psm_p", min_sig_months=min_sig_months)
    naive_c = monthly_consistency(ren, "naive_diff", "naive_p", min_sig_months=min_sig_months)
    aha_c = monthly_consistency(aha, "strat_ate", "strat_p", min_sig_months=min_sig_months)

    rows = []
    for feat, g in ren.groupby("feature", sort=False):
        a = aha[aha["feature"] == feat]
        att, lo, hi = _pooled_ci(g, "psm_att", "psm_se")
        row = {
            "feature": feat,
            "adoption_rate": g["adoption_rate"].mean(),
            "naive_diff": g["naive_diff"].mean(),
            "psm_att": att,
            "psm_ci_low": lo,
            "psm_ci_high": hi,
            "psm_lift": g["psm_lift"].mean(),
            "max_abs_smd_after": g["max_abs_smd_after"].max(),
            "aha_effect": a["strat_ate"].mean(),
            "aha_psm_att": a["psm_att"].mean(),
            "aha_interpretable_months": int(a["strat_interpretable"].sum()),
        }
        if "true_att" in g:
            row["true_att"] = g["true_att"].mean()
            row["aha_true_att"] = a["true_att"].mean()
        rows.append(row)
    t = pd.DataFrame(rows)
    t = t.merge(psm_c.drop(columns=["mean_effect", "n_months"]), on="feature")
    t = t.merge(
        naive_c[["feature", "confirmed", "months_significant"]].rename(
            columns={"confirmed": "naive_confirmed", "months_significant": "naive_months_significant"}
        ),
        on="feature",
    )
    t = t.merge(
        aha_c[["feature", "confirmed", "months_significant", "trend"]].rename(
            columns={"confirmed": "aha_confirmed", "months_significant": "aha_months_significant", "trend": "aha_trend"}
        ),
        on="feature",
    )
    # a trend on a non-confirmed effect is just noise around zero - do not label it
    t.loc[~t["confirmed"], "trend"] = "n/a"
    t.loc[~t["aha_confirmed"], "aha_trend"] = "n/a"
    t = assign_quadrants(t, impact_col="psm_att", confirmed_col="confirmed")
    return t


METHODS = {
    "Naive (users vs non-users)": ("naive_diff", None, None),
    "PSM (1:1 NN, caliper 0.2 SD)": ("psm_att", "psm_ci_low", "psm_ci_high"),
    "Stratified (skip zero-var strata)": ("strat_ate", "strat_ci_low", "strat_ci_high"),
    "Stratified (keep zero-var strata)": ("strat_keepzv_ate", "strat_keepzv_ci_low", "strat_keepzv_ci_high"),
}
_P_COLS = {"naive_diff": "naive_p", "psm_att": "psm_p", "strat_ate": "strat_p", "strat_keepzv_ate": "strat_keepzv_p"}


def evaluate(monthly: pd.DataFrame, min_sig_months: int = 4) -> pd.DataFrame:
    """Bias / MAE / RMSE / CI coverage vs ground truth over all feature x month estimates.

    Also counts, at the feature level, truly-null features wrongly *confirmed*
    (false discoveries), truly non-null features not confirmed (misses), and
    non-null features confirmed with the wrong sign.
    """
    rows = []
    for outcome, g in monthly.groupby("outcome", sort=False):
        null = g.groupby("feature")["true_logit_effect"].apply(lambda s: bool((s == 0).all()))
        true_sign = np.sign(g.groupby("feature")["true_att"].mean())
        for name, (est, lo, hi) in METHODS.items():
            err = (g[est] - g["true_att"]).dropna()
            row = {
                "outcome": outcome,
                "method": name,
                "n_estimates": int(len(err)),
                "bias_pp": 100 * err.mean(),
                "mae_pp": 100 * err.abs().mean(),
                "rmse_pp": 100 * np.sqrt((err**2).mean()),
                "ci_coverage": (
                    float(((g[lo] <= g["true_att"]) & (g["true_att"] <= g[hi])).mean()) if lo else np.nan
                ),
                "false_confirmed": np.nan,
                "missed": np.nan,
                "wrong_sign": np.nan,
            }
            if est in _P_COLS:
                c = monthly_consistency(g, est, _P_COLS[est], min_sig_months=min_sig_months).set_index("feature")
                c = c.join(null.rename("is_null"))
                row["false_confirmed"] = int((c["confirmed"] & c["is_null"]).sum())
                row["missed"] = int((~c["confirmed"] & ~c["is_null"]).sum())
                sign = c["direction"].map({"positive": 1.0, "negative": -1.0, "none": 0.0})
                wrong = c["confirmed"] & ~c["is_null"] & (sign != true_sign.reindex(c.index))
                row["wrong_sign"] = int(wrong.sum())
            rows.append(row)
    return pd.DataFrame(rows)
