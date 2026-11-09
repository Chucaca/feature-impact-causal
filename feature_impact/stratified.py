"""Stratified (exact-cell) treatment effect, used for the AHA outcome.

Strata = industry x city_tier x contract_type x gmv_tier x invoice_tier. Inside
each stratum merchants are comparable on those confounders, so the within-
stratum difference in AHA rate between users and non-users is a local
effect estimate. Strata effects are averaged with weights = number of treated
merchants in the stratum, so the estimand targets the treated population
(ATT-like), consistent with the PSM estimate.

Strata are skipped when

* they have fewer than ``min_rows`` rows (default 5),
* they have no treated or no control merchants (no comparison possible),
* the control outcome has zero variance (e.g. every control reached AHA, or
  none did). Such a stratum's "effect" is driven by a degenerate baseline;
  in particular a stratum where ``outcome_control == 0`` reports the full
  treated rate as "effect", which is not a causal quantity.

The result is flagged ``interpretable=False`` when too few strata survive or
the surviving strata cover less than ``min_coverage`` of the treated
merchants - then the number describes a non-representative subset only.

Note: skipping zero-variance strata is a deliberate, documented trade-off. It
removes degenerate comparisons but can bias the estimate (strata where every
control renewed tend to have small or negative true lifts). The evaluation
quantifies this against the ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StratifiedResult:
    treatment: str
    outcome: str
    ate: float = np.nan
    se: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    p_value: float = np.nan
    n_strata_total: int = 0
    n_strata_used: int = 0
    n_skipped_small: int = 0
    n_skipped_one_arm: int = 0
    n_skipped_zero_var: int = 0
    treated_coverage: float = 0.0  # share of treated rows inside used strata
    interpretable: bool = False
    note: str = ""

    def to_row(self) -> dict:
        return dict(self.__dict__)


def stratum_table(df: pd.DataFrame, treatment: str, outcome: str, strata_cols: list[str]) -> pd.DataFrame:
    """Per-stratum counts, means and variances of the outcome by arm."""
    g = df.groupby(strata_cols + [treatment], observed=True)[outcome]
    agg = g.agg(n="size", mean="mean", var=lambda s: s.var(ddof=0)).unstack(treatment)
    agg.columns = [f"{stat}_{'t' if arm == 1 else 'c'}" for stat, arm in agg.columns]
    for col in ("n_t", "n_c"):
        if col not in agg:
            agg[col] = 0
        agg[col] = agg[col].fillna(0).astype(int)
    for col in ("mean_t", "mean_c", "var_t", "var_c"):
        if col not in agg:
            agg[col] = np.nan
    agg["n"] = agg["n_t"] + agg["n_c"]
    return agg.reset_index()


def stratified_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    strata_cols: list[str],
    min_rows: int = 5,
    min_strata: int = 3,
    min_coverage: float = 0.5,
    skip_zero_var: bool = True,
) -> StratifiedResult:
    """Treated-weighted stratified difference in means with skip rules and an interpretability flag."""
    res = StratifiedResult(treatment=treatment, outcome=outcome)
    tab = stratum_table(df, treatment, outcome, strata_cols)
    res.n_strata_total = len(tab)

    small = tab["n"] < min_rows
    one_arm = ~small & ((tab["n_t"] == 0) | (tab["n_c"] == 0))
    zero_var = ~small & ~one_arm & (tab["var_c"].fillna(0) == 0) & skip_zero_var
    used = tab[~small & ~one_arm & ~zero_var]
    res.n_skipped_small, res.n_skipped_one_arm = int(small.sum()), int(one_arm.sum())
    res.n_skipped_zero_var, res.n_strata_used = int(zero_var.sum()), len(used)

    total_treated = int(tab["n_t"].sum())
    res.treated_coverage = float(used["n_t"].sum() / total_treated) if total_treated else 0.0

    if len(used) == 0:
        res.note = "no usable strata"
        return res

    w = used["n_t"].to_numpy(dtype=float)
    eff = (used["mean_t"] - used["mean_c"]).to_numpy(dtype=float)
    res.ate = float(np.sum(w * eff) / w.sum())
    # variance of a weighted sum of independent stratum differences (Bernoulli-style ddof=0 variances)
    v = (used["var_t"] / used["n_t"] + used["var_c"] / used["n_c"]).to_numpy(dtype=float)
    res.se = float(np.sqrt(np.sum(w**2 * v)) / w.sum())
    res.ci_low, res.ci_high = res.ate - 1.96 * res.se, res.ate + 1.96 * res.se
    res.p_value = float(2 * stats.norm.sf(abs(res.ate) / res.se)) if res.se > 0 else np.nan

    problems = []
    if len(used) < min_strata:
        problems.append(f"only {len(used)} usable strata (<{min_strata})")
    if res.treated_coverage < min_coverage:
        problems.append(f"usable strata cover {res.treated_coverage:.0%} of treated (<{min_coverage:.0%})")
    res.interpretable = not problems
    res.note = "; ".join(problems)
    return res
