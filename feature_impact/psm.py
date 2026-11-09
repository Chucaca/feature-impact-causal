"""Propensity score matching (PSM) for the ATT of one feature on one outcome.

Design choices (documented on purpose - these are the knobs a reviewer asks about):

* **Propensity model** - ``sklearn`` ``LogisticRegression`` on one-hot encoded
  categorical confounders plus standardised numeric confounders. Light L2
  regularisation (``C=1.0``) keeps the fit stable when a one-hot level is rare.
* **Matching metric** - 1:1 nearest neighbour on the *logit* of the propensity
  score (``sklearn.neighbors.NearestNeighbors``). The logit spreads out scores
  near 0 and 1, where raw-probability distances are compressed.
* **Caliper** - default ``0.2 * SD(logit propensity)`` (Austin, 2011), which
  scales with how separable treated and control are. A fixed absolute caliper
  (e.g. ``0.05`` on the logit) can be passed instead. Treated units with no
  control inside the caliper are dropped and reported (``n_unmatched``) - this
  is the overlap / common-support guard.
* **Replacement** - default *with* replacement: each treated unit gets its
  closest control even if that control was used before. This minimises bias
  (best match always) at the cost of some variance, and makes results
  independent of matching order. ``replace=False`` runs greedy matching
  (hardest-to-match treated first) for comparison.
* **Inference** - ATT = mean of matched pair differences. 95% CI from a
  percentile *unit-level* Poisson bootstrap conditional on the match: every
  treated unit and every unique control gets an independent Poisson(1)
  weight, so a control re-used k times moves k pairs at once. (A naive
  pair bootstrap treats re-used controls as independent and under-covers -
  ~86% instead of 95% on the synthetic benchmark.) The p-value is a
  two-sided normal test with the bootstrap standard error. Weighting the two
  arms independently ignores the positive correlation inside a pair, so the
  interval is mildly conservative (~98% empirical coverage for a nominal
  95%); uncertainty from the propensity fit itself is ignored. Coverage is
  reported in the evaluation.
* **Balance** - standardised mean difference (SMD) of every covariate before
  and after matching, using the pre-matching pooled SD as denominator so the
  two numbers are comparable. ``|SMD| > 0.1`` is flagged as imbalanced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

MIN_TREATED = 50
MIN_CONTROL = 50
SMD_THRESHOLD = 0.1


@dataclass
class PSMResult:
    treatment: str
    outcome: str
    status: str  # "ok" | "insufficient_sample" | "no_matches"
    n_treated: int = 0
    n_control: int = 0
    n_matched: int = 0
    n_unmatched: int = 0
    n_unique_controls: int = 0
    caliper: float = np.nan
    att: float = np.nan
    treated_mean: float = np.nan
    control_mean: float = np.nan
    lift: float = np.nan
    se: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    p_value: float = np.nan
    naive_diff: float = np.nan
    max_abs_smd_before: float = np.nan
    max_abs_smd_after: float = np.nan
    n_imbalanced_after: int = 0
    balance: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def to_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "balance"}
        d["balanced"] = bool(self.n_imbalanced_after == 0) if self.status == "ok" else False
        return d


def design_matrix(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    """One-hot categorical confounders (all levels kept, for balance reporting) + raw numerics."""
    parts = []
    if cat_cols:
        parts.append(pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols, prefix_sep="=", dtype=float))
    if num_cols:
        parts.append(df[num_cols].astype(float))
    return pd.concat(parts, axis=1)


def fit_propensity(X: pd.DataFrame, t: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return propensity scores P(T=1 | X) from a logistic regression."""
    Xs = X.to_numpy(dtype=float)
    mu, sd = Xs.mean(axis=0), Xs.std(axis=0)
    sd[sd == 0] = 1.0
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    model.fit((Xs - mu) / sd, t)
    return model.predict_proba((Xs - mu) / sd)[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def smd_table(X: pd.DataFrame, t: np.ndarray, treated_idx: np.ndarray, control_idx: np.ndarray) -> pd.DataFrame:
    """SMD before (all units) and after (matched units, duplicates counted) matching."""
    Xv = X.to_numpy(dtype=float)
    xt, xc = Xv[t == 1], Xv[t == 0]
    pooled_sd = np.sqrt((xt.var(axis=0, ddof=1) + xc.var(axis=0, ddof=1)) / 2.0)
    pooled_sd[pooled_sd == 0] = np.nan
    before = (xt.mean(axis=0) - xc.mean(axis=0)) / pooled_sd
    after = (Xv[treated_idx].mean(axis=0) - Xv[control_idx].mean(axis=0)) / pooled_sd
    out = pd.DataFrame({"covariate": X.columns, "smd_before": before, "smd_after": after})
    out["imbalanced_before"] = out["smd_before"].abs() > SMD_THRESHOLD
    out["imbalanced_after"] = out["smd_after"].abs() > SMD_THRESHOLD
    return out


def _match(lt: np.ndarray, lc: np.ndarray, caliper: float, replace: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (treated positions, control positions) of the matched pairs."""
    if replace:
        nn = NearestNeighbors(n_neighbors=1).fit(lc.reshape(-1, 1))
        dist, idx = nn.kneighbors(lt.reshape(-1, 1))
        ok = dist[:, 0] <= caliper
        return np.flatnonzero(ok), idx[ok, 0]

    # greedy without replacement: hardest-to-match (highest propensity) treated first
    k = min(len(lc), 50)
    nn = NearestNeighbors(n_neighbors=k).fit(lc.reshape(-1, 1))
    dist, idx = nn.kneighbors(lt.reshape(-1, 1))
    used = np.zeros(len(lc), dtype=bool)
    t_pos, c_pos = [], []
    for i in np.argsort(-lt):
        for d, j in zip(dist[i], idx[i]):
            if d > caliper:
                break
            if not used[j]:
                used[j] = True
                t_pos.append(i)
                c_pos.append(j)
                break
    return np.asarray(t_pos, dtype=int), np.asarray(c_pos, dtype=int)


def _unit_bootstrap(y_t: np.ndarray, y_c: np.ndarray, c_ids: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Poisson bootstrap that resamples *units*, not pairs.

    With replacement a control can serve several treated units. Resampling
    pairs would treat those re-uses as independent and understate the
    variance, so each treated unit and each *unique* control gets its own
    Poisson(1) weight; a control's weight scales all of its re-uses together.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(c_ids, return_inverse=True)
    k = np.bincount(inv)  # times each unique control is used
    yc_u = np.bincount(inv, weights=y_c)  # sum of outcomes over re-uses (= k * y)
    out = np.empty(n_boot)
    chunk = max(1, int(2e5 // max(len(y_t), len(uniq))))  # keep peak memory small
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        wt = rng.poisson(1.0, size=(b, len(y_t))).astype(np.float32)
        wc = rng.poisson(1.0, size=(b, len(uniq))).astype(np.float32)
        mt = (wt @ y_t) / np.maximum(wt.sum(axis=1), 1)
        mc = (wc @ yc_u) / np.maximum(wc @ k, 1)
        out[s : s + b] = mt - mc
    return out


def estimate_att(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    cat_cols: list[str],
    num_cols: list[str],
    caliper: float | str = "0.2sd",
    replace: bool = True,
    n_boot: int = 500,
    seed: int = 0,
    min_treated: int = MIN_TREATED,
    min_control: int = MIN_CONTROL,
) -> PSMResult:
    """Estimate the ATT of ``treatment`` on ``outcome`` by 1:1 propensity score matching."""
    t = df[treatment].to_numpy(dtype=int)
    y = df[outcome].to_numpy(dtype=float)
    n_t, n_c = int(t.sum()), int((1 - t).sum())
    res = PSMResult(treatment=treatment, outcome=outcome, status="ok", n_treated=n_t, n_control=n_c)
    if n_t < min_treated or n_c < min_control:
        res.status = "insufficient_sample"
        return res
    res.naive_diff = float(y[t == 1].mean() - y[t == 0].mean())

    X = design_matrix(df, cat_cols, num_cols)
    lp = _logit(fit_propensity(X, t, seed=seed))
    if caliper == "0.2sd":
        cal = 0.2 * float(lp.std(ddof=1))
    else:
        cal = float(caliper)
    res.caliper = cal

    treated_rows, control_rows = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    tp, cp = _match(lp[treated_rows], lp[control_rows], cal, replace)
    res.n_matched, res.n_unmatched = len(tp), n_t - len(tp)
    if len(tp) < min_treated:
        res.status = "no_matches"
        return res
    ti, ci = treated_rows[tp], control_rows[cp]
    res.n_unique_controls = int(len(np.unique(ci)))

    diffs = y[ti] - y[ci]
    res.att = float(diffs.mean())
    res.treated_mean, res.control_mean = float(y[ti].mean()), float(y[ci].mean())
    res.lift = res.att / res.control_mean if res.control_mean > 0 else np.nan

    boot = _unit_bootstrap(y[ti], y[ci], ci, n_boot, seed)
    res.se = float(boot.std(ddof=1))
    res.ci_low, res.ci_high = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    res.p_value = float(2 * stats.norm.sf(abs(res.att) / res.se)) if res.se > 0 else (0.0 if res.att else 1.0)

    bal = smd_table(X, t, ti, ci)
    res.balance = bal
    res.max_abs_smd_before = float(bal["smd_before"].abs().max())
    res.max_abs_smd_after = float(bal["smd_after"].abs().max())
    res.n_imbalanced_after = int(bal["imbalanced_after"].sum())
    return res


def naive_diff(df: pd.DataFrame, treatment: str, outcome: str) -> dict:
    """Naive users-vs-non-users difference in means with a two-proportion z-test."""
    t = df[treatment].to_numpy(dtype=int)
    y = df[outcome].to_numpy(dtype=float)
    y1, y0 = y[t == 1], y[t == 0]
    if len(y1) == 0 or len(y0) == 0:
        return {"naive_diff": np.nan, "naive_se": np.nan, "naive_p": np.nan}
    diff = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    p = 2 * stats.norm.sf(abs(diff) / se) if se > 0 else np.nan
    return {"naive_diff": float(diff), "naive_se": float(se), "naive_p": float(p)}
