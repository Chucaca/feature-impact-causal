"""Month-over-month consistency: one significant month is not a finding.

The estimators run independently on each monthly snapshot (each snapshot is
a different renewal cohort). A feature is **confirmed** only if its effect is
significant (p < alpha) in at least ``min_sig_months`` of the snapshots
(default 4 of 6) *with a consistent sign*. With 6 independent looks at a
truly null feature, P(>=4 false positives at alpha=0.05) < 1e-4 per feature, so
this rule is far stricter than a single test and removes month-to-month
noise.

The trend label comes from the OLS slope of the monthly estimate over the
last ``trend_window`` snapshots (default 3): "rising" / "declining" if
|slope| exceeds ``slope_tol`` (default 0.01 = 1 pp per month),
otherwise "stable".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trend_label(values: np.ndarray, slope_tol: float = 0.01) -> tuple[float, str]:
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return np.nan, "insufficient"
    slope = float(np.polyfit(np.arange(len(vals)), vals, 1)[0])
    if slope > slope_tol:
        return slope, "rising"
    if slope < -slope_tol:
        return slope, "declining"
    return slope, "stable"


def monthly_consistency(
    monthly: pd.DataFrame,
    est_col: str,
    p_col: str,
    feature_col: str = "feature",
    month_col: str = "snapshot",
    alpha: float = 0.05,
    min_sig_months: int = 4,
    trend_window: int = 3,
    slope_tol: float = 0.01,
) -> pd.DataFrame:
    """Summarise per-feature monthly estimates into confirmed / trend labels."""
    rows = []
    for feat, g in monthly.sort_values(month_col).groupby(feature_col, sort=False):
        est = g[est_col].to_numpy(dtype=float)
        p = g[p_col].to_numpy(dtype=float)
        sig = (p < alpha) & ~np.isnan(est)
        n_pos, n_neg = int((sig & (est > 0)).sum()), int((sig & (est < 0)).sum())
        n_sig = max(n_pos, n_neg)
        direction = "positive" if n_pos >= n_neg and n_pos > 0 else ("negative" if n_neg > 0 else "none")
        slope, label = trend_label(est[-trend_window:], slope_tol)
        rows.append(
            {
                feature_col: feat,
                "n_months": int((~np.isnan(est)).sum()),
                "months_significant": n_sig,
                "direction": direction,
                "confirmed": n_sig >= min_sig_months,
                "mean_effect": float(np.nanmean(est)) if (~np.isnan(est)).any() else np.nan,
                "trend_slope": slope,
                "trend": label,
            }
        )
    return pd.DataFrame(rows)
