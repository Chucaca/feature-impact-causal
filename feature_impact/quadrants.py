"""Adoption x impact quadrants - turning estimates into a product decision.

* **Champion**                 - high adoption, high impact: protect & keep investing.
* **Hidden gem**               - low adoption, high impact: onboarding opportunity
  (push it in the first weeks, in-app nudges, sales playbooks).
* **Widely used, low impact**  - popular but does not move retention: hygiene
  feature, do not over-invest; do not market it as a retention driver.
* **Low / low**                - low adoption, low impact: deprioritise or rethink.

"High impact" requires a *confirmed* (consistent across months), positive
effect at least ``impact_threshold`` (probability points). A confirmed
negative effect is low impact and additionally flagged ``harmful``.
"""
from __future__ import annotations

import pandas as pd

CHAMPION = "Champion"
HIDDEN_GEM = "Hidden gem"
WIDE_LOW = "Widely used, low impact"
LOW_LOW = "Low / low"
QUADRANTS = (CHAMPION, HIDDEN_GEM, WIDE_LOW, LOW_LOW)


def classify(
    adoption: float,
    impact: float,
    confirmed: bool,
    adoption_threshold: float,
    impact_threshold: float = 0.02,
) -> str:
    high_adoption = adoption >= adoption_threshold
    high_impact = bool(confirmed) and impact >= impact_threshold
    if high_adoption and high_impact:
        return CHAMPION
    if high_impact:
        return HIDDEN_GEM
    if high_adoption:
        return WIDE_LOW
    return LOW_LOW


def assign_quadrants(
    df: pd.DataFrame,
    adoption_col: str = "adoption_rate",
    impact_col: str = "psm_att",
    confirmed_col: str = "confirmed",
    adoption_threshold: float | None = None,
    impact_threshold: float = 0.02,
) -> pd.DataFrame:
    """Add ``quadrant`` and ``harmful`` columns. Default adoption threshold = median adoption."""
    out = df.copy()
    thr = float(out[adoption_col].median()) if adoption_threshold is None else adoption_threshold
    out["quadrant"] = [
        classify(a, i, c, thr, impact_threshold)
        for a, i, c in zip(out[adoption_col], out[impact_col], out[confirmed_col])
    ]
    out["harmful"] = out[confirmed_col].astype(bool) & (out[impact_col] <= -impact_threshold)
    out.attrs["adoption_threshold"] = thr
    out.attrs["impact_threshold"] = impact_threshold
    return out
