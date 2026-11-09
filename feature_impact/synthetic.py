"""Seeded synthetic data generator with a KNOWN causal ground truth.

Setting: a point-of-sale (POS) SaaS product with ~50 features (we simulate 20 of
them); merchants renew yearly. Every monthly snapshot contains the cohort of
merchants whose renewal falls in that month.

For each merchant-month we generate

* confounders  - industry, city_tier, contract_type, gmv_tier, invoice_tier,
  branches, tenure_years. Bigger merchants (more branches, higher GMV and
  invoice volume, longer contracts) both adopt more features AND renew more.
* treatments   - ``used_<feature>`` = 1 if the feature was used at least once
  in months T-4..T-2 before the renewal month T. Adoption probability is a
  logistic function of the confounders (plus an industry affinity).
* outcomes     - ``renewed`` (renewed the yearly contract at T) and ``aha``
  (reached the "AHA moment": still active long-term). Both are Bernoulli
  draws from a logistic model of the confounders plus a known causal effect
  for each feature.

Because the outcome model is known, the true average treatment effect on the
treated (ATT, probability scale) is computed exactly for every
feature x month x outcome as ``mean(p(Y | T=1) - p(Y | T=0))`` over the treated
merchants, holding everything else fixed. That is the ground truth every
estimator is graded against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

INDUSTRIES: tuple[str, ...] = ("fnb", "fashion", "grocery", "pharmacy", "electronics")
INDUSTRY_P: tuple[float, ...] = (0.30, 0.22, 0.25, 0.10, 0.13)
CONTRACTS: tuple[str, ...] = ("monthly", "annual", "multi_year")

CAT_CONFOUNDERS: list[str] = ["industry", "city_tier", "contract_type", "gmv_tier", "invoice_tier"]
NUM_CONFOUNDERS: list[str] = ["log_branches", "tenure_years"]
STRATA_COLS: list[str] = list(CAT_CONFOUNDERS)
OUTCOMES: tuple[str, ...] = ("renewed", "aha")


@dataclass(frozen=True)
class FeatureSpec:
    """One product feature and its ground-truth causal effects (logit scale)."""

    name: str
    adopt_intercept: float
    size_slope: float  # how much more likely bigger merchants are to adopt
    renew_effect: float  # causal effect on renewal (logit scale)
    aha_effect: float  # causal effect on AHA (logit scale)
    drift: float = 0.0  # change of renew_effect per month (centred on the middle month)
    affinity: tuple[tuple[str, float], ...] = field(default_factory=tuple)  # industry -> logit bonus


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("loyalty_points", -0.4, 0.9, 0.40, 0.25),
    FeatureSpec("inventory_alerts", 0.3, 0.6, 0.45, 0.30),
    FeatureSpec("e_invoice", 0.2, 1.2, 0.00, 0.00),
    FeatureSpec("multi_branch_transfer", -2.0, 1.8, 0.00, 0.00),
    FeatureSpec("promotions_engine", -0.6, 0.8, 0.30, 0.20),
    FeatureSpec("mobile_app", 0.5, 0.5, 0.25, 0.35),
    FeatureSpec("online_store_sync", -2.4, 0.5, 0.60, 0.40),
    FeatureSpec("auto_reorder", -2.6, 0.6, 0.70, 0.30),
    FeatureSpec("customer_sms", -1.0, 0.9, 0.00, 0.00),
    FeatureSpec("staff_permissions", -0.8, 1.4, 0.00, 0.10),
    FeatureSpec("shift_reports", -0.3, 0.7, 0.20, 0.15),
    FeatureSpec("barcode_printing", 0.0, 0.6, 0.00, 0.00),
    FeatureSpec("debt_tracking", -0.9, 0.3, 0.35, 0.45),
    FeatureSpec("delivery_integration", -1.6, 0.7, 0.30, 0.25, drift=0.08),
    FeatureSpec("table_management", -2.0, 0.4, 0.00, 0.00, affinity=(("fnb", 2.5),)),
    FeatureSpec("kitchen_display", -2.6, 0.8, 0.30, 0.20, affinity=(("fnb", 2.0),)),
    FeatureSpec("batch_expiry", -2.5, 0.5, 0.00, 0.15, affinity=(("pharmacy", 2.8), ("grocery", 1.0))),
    FeatureSpec("price_books", -1.2, 1.0, 0.00, 0.00),
    FeatureSpec("legacy_desktop_sync", -1.8, 0.6, -0.40, -0.30),
    FeatureSpec("analytics_dashboard", -0.7, 1.0, 0.25, 0.20, drift=-0.06),
)
FEATURE_NAMES: list[str] = [f.name for f in FEATURES]


def treatment_col(feature: str) -> str:
    return f"used_{feature}"


@dataclass
class SyntheticData:
    df: pd.DataFrame  # one row per merchant-month
    truth: pd.DataFrame  # snapshot, feature, outcome, true_att, true_ate
    features: tuple[FeatureSpec, ...]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def renew_effect_at(spec: FeatureSpec, snapshot: int, n_months: int) -> float:
    """Ground-truth logit effect on renewal in a given snapshot (1-based)."""
    centre = (n_months + 1) / 2.0
    return spec.renew_effect + spec.drift * (snapshot - centre)


def _confounders(rng: np.random.Generator, n: int) -> pd.DataFrame:
    z = rng.normal(size=n)  # latent merchant scale (NOT observed by analysts)
    industry = rng.choice(np.array(INDUSTRIES), size=n, p=np.array(INDUSTRY_P))
    branches = 1 + rng.poisson(np.exp(0.9 * z - 0.7))
    gmv_tier = np.digitize(0.8 * z + rng.normal(0, 0.6, n), [-0.8, 0.0, 0.8]) + 1
    invoice_tier = np.digitize(0.7 * z + rng.normal(0, 0.7, n), [-0.8, 0.0, 0.8]) + 1

    c_logits = np.column_stack([np.zeros(n), 0.3 + 0.6 * z, -1.2 + 0.9 * z])
    c_prob = np.exp(c_logits) / np.exp(c_logits).sum(axis=1, keepdims=True)
    contract_idx = (rng.random(n)[:, None] > np.cumsum(c_prob, axis=1)).sum(axis=1)
    contract = np.array(CONTRACTS)[np.minimum(contract_idx, 2)]

    t_logits = np.column_stack([0.3 * z, np.zeros(n), -0.3 * z])
    t_prob = np.exp(t_logits) / np.exp(t_logits).sum(axis=1, keepdims=True)
    city_idx = (rng.random(n)[:, None] > np.cumsum(t_prob, axis=1)).sum(axis=1)
    city_tier = np.minimum(city_idx, 2) + 1

    tenure = np.clip(rng.gamma(2.0, 1.2, n) + 0.3 * np.maximum(z, 0), 0.1, 10.0).round(1)

    return pd.DataFrame(
        {
            "industry": industry,
            "city_tier": city_tier.astype(int),
            "contract_type": contract,
            "gmv_tier": gmv_tier.astype(int),
            "invoice_tier": invoice_tier.astype(int),
            "branches": branches.astype(int),
            "log_branches": np.log1p(branches),
            "tenure_years": tenure,
        }
    )


def _size_index(c: pd.DataFrame) -> np.ndarray:
    """Observed 'merchant scale' index that drives feature adoption."""
    return (
        0.45 * (c["gmv_tier"] - 2.5)
        + 0.35 * (c["invoice_tier"] - 2.5)
        + 0.60 * (c["log_branches"] - 0.8)
        + 0.40 * (c["contract_type"] == "annual")
        + 0.70 * (c["contract_type"] == "multi_year")
        + 0.05 * (c["tenure_years"] - 2.5)
    ).to_numpy(dtype=float)


_IND_RENEW = {"fnb": -0.20, "fashion": -0.10, "grocery": 0.10, "pharmacy": 0.25, "electronics": 0.0}
_CONTRACT_RENEW = {"monthly": -0.30, "annual": 0.20, "multi_year": 0.50}


def _base_logits(c: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ind = c["industry"].map(_IND_RENEW).to_numpy(dtype=float)
    con = c["contract_type"].map(_CONTRACT_RENEW).to_numpy(dtype=float)
    city = np.where(c["city_tier"] == 1, 0.10, np.where(c["city_tier"] == 3, -0.10, 0.0))
    common = (
        0.35 * (c["gmv_tier"] - 2.5)
        + 0.25 * (c["invoice_tier"] - 2.5)
        + 0.40 * (c["log_branches"] - 0.8)
        + 0.08 * np.minimum(c["tenure_years"], 6.0)
    ).to_numpy(dtype=float)
    renew = -0.9 + ind + con + city + common
    aha = -1.1 + 0.6 * ind + 0.8 * con + city + 1.1 * common
    return renew, aha


def generate(n_per_month: int = 10_000, n_months: int = 6, seed: int = 42) -> SyntheticData:
    """Generate ``n_per_month * n_months`` merchant-month rows plus the ground truth."""
    rng = np.random.default_rng(seed)
    frames, truth_rows = [], []
    for snap in range(1, n_months + 1):
        c = _confounders(rng, n_per_month)
        size = _size_index(c)
        renew_logit, aha_logit = _base_logits(c)
        treat: dict[str, np.ndarray] = {}
        betas: dict[str, tuple[float, float]] = {}
        for spec in FEATURES:
            aff = dict(spec.affinity)
            bonus = c["industry"].map(lambda i, aff=aff: aff.get(i, 0.0)).to_numpy(dtype=float)
            logit = spec.adopt_intercept + spec.size_slope * size + bonus + 0.15 * (c["city_tier"] == 1)
            t = (rng.random(n_per_month) < _sigmoid(np.asarray(logit, dtype=float))).astype(int)
            b_r, b_a = renew_effect_at(spec, snap, n_months), spec.aha_effect
            renew_logit = renew_logit + b_r * t
            aha_logit = aha_logit + b_a * t
            treat[spec.name], betas[spec.name] = t, (b_r, b_a)

        p_renew, p_aha = _sigmoid(renew_logit), _sigmoid(aha_logit)
        renewed = (rng.random(n_per_month) < p_renew).astype(int)
        aha = (rng.random(n_per_month) < p_aha).astype(int)

        # exact ground truth from the known outcome model
        for name, t in treat.items():
            for outcome, logit, beta in (
                ("renewed", renew_logit, betas[name][0]),
                ("aha", aha_logit, betas[name][1]),
            ):
                l0 = logit - beta * t
                diff = _sigmoid(l0 + beta) - _sigmoid(l0)
                truth_rows.append(
                    {
                        "snapshot": snap,
                        "feature": name,
                        "outcome": outcome,
                        "true_att": float(diff[t == 1].mean()) if t.any() else np.nan,
                        "true_ate": float(diff.mean()),
                        "true_logit_effect": beta,
                    }
                )

        frame = c.copy()
        frame.insert(0, "merchant_id", [f"m{snap}_{i:06d}" for i in range(n_per_month)])
        frame.insert(1, "snapshot", snap)
        for name, t in treat.items():
            frame[treatment_col(name)] = t
        frame["renewed"] = renewed
        frame["aha"] = aha
        frames.append(frame)

    return SyntheticData(
        df=pd.concat(frames, ignore_index=True),
        truth=pd.DataFrame(truth_rows),
        features=FEATURES,
    )
