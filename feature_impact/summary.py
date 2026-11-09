"""Lifecycle-stage summaries (Onboarding / Active / Renewal) for non-technical readers.

The numbers are always computed deterministically; an LLM is only used to
*word* the summary, constrained by a Pydantic schema (``instructor``), so it
cannot invent fields. Provider is chosen by environment variables:

* ``FEATURE_IMPACT_LLM``        - ``gemini`` | ``openai`` | ``none`` (default ``none``)
* ``FEATURE_IMPACT_LLM_MODEL``  - optional model name override
* ``GEMINI_API_KEY`` / ``OPENAI_API_KEY`` - provider credentials

All provider imports are lazy. If no provider is configured, a package is
missing, or the call fails, a deterministic template summary is produced,
so the pipeline always runs offline.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

Stage = Literal["Onboarding", "Active", "Renewal"]
STAGES: tuple[str, ...] = ("Onboarding", "Active", "Renewal")
DEFAULT_MODELS = {"gemini": "gemini-2.0-flash", "openai": "gpt-4o-mini"}


class FeatureCallout(BaseModel):
    feature: str
    effect_pp: float = Field(description="Estimated effect in percentage points")
    evidence: str = Field(description="One sentence: method, months significant, CI")


class StageSummary(BaseModel):
    stage: Stage
    headline: str = Field(description="One-sentence takeaway for a product manager")
    key_features: list[FeatureCallout] = Field(default_factory=list, max_length=5)
    actions: list[str] = Field(default_factory=list, max_length=4)
    caveats: list[str] = Field(default_factory=list, max_length=4)


class FeatureImpactSummary(BaseModel):
    stages: list[StageSummary]
    data_note: str


# ----------------------------------------------------------------------------- facts
def build_facts(table: pd.DataFrame) -> dict:
    """Compact, LLM-safe facts per stage (no raw data, only aggregates)."""

    def rows(df: pd.DataFrame, eff: str, extra: list[str]) -> list[dict]:
        cols = ["feature", eff, *extra]
        return json.loads(df[cols].round(4).to_json(orient="records"))

    t = table
    onboarding = t[t["aha_confirmed"] & (t["aha_effect"] > 0)].sort_values("aha_effect", ascending=False)
    renewal = t[t["confirmed"] & (t["psm_att"] > 0)].sort_values("psm_att", ascending=False)
    harmful = t[t["harmful"]]
    return {
        "Onboarding": {
            "outcome": "AHA moment (long-term active), stratified estimate, confirmed in >=4/6 months",
            "drivers": rows(onboarding.head(5), "aha_effect", ["aha_months_significant", "adoption_rate"]),
            "hidden_gems": t.loc[t["quadrant"] == "Hidden gem", "feature"].tolist(),
        },
        "Active": {
            "outcome": "adoption vs renewal impact quadrants and monthly trends",
            "champions": t.loc[t["quadrant"] == "Champion", "feature"].tolist(),
            "widely_used_low_impact": t.loc[t["quadrant"] == "Widely used, low impact", "feature"].tolist(),
            "rising": t.loc[t["trend"] == "rising", "feature"].tolist(),
            "declining": t.loc[t["trend"] == "declining", "feature"].tolist(),
        },
        "Renewal": {
            "outcome": "renewal, PSM ATT, confirmed in >=4/6 months",
            "drivers": rows(renewal.head(5), "psm_att", ["months_significant", "psm_ci_low", "psm_ci_high"]),
            "harmful": rows(harmful, "psm_att", ["months_significant"]),
            "naive_false_positives": t.loc[t["naive_confirmed"] & ~t["confirmed"], "feature"].tolist(),
        },
    }


# ----------------------------------------------------------------------------- template
def _pp(x: float) -> str:
    return f"{x * 100:+.1f} pp"


def template_summary(table: pd.DataFrame) -> FeatureImpactSummary:
    """Deterministic, offline summary built from the same facts."""
    t = table
    stages: list[StageSummary] = []

    onb = t[t["aha_confirmed"] & (t["aha_effect"] > 0)].sort_values("aha_effect", ascending=False).head(5)
    gems = t.loc[t["quadrant"] == "Hidden gem", "feature"].tolist()
    stages.append(
        StageSummary(
            stage="Onboarding",
            headline=(
                f"{len(onb)} features consistently raise the chance of reaching the AHA moment; "
                f"{len(gems)} high-impact features are still used by few merchants."
            ),
            key_features=[
                FeatureCallout(
                    feature=r.feature,
                    effect_pp=round(r.aha_effect * 100, 1),
                    evidence=f"stratified estimate {_pp(r.aha_effect)}, significant in {r.aha_months_significant}/6 months",
                )
                for r in onb.itertuples()
            ],
            actions=[f"Put {', '.join(gems)} into the first-30-day onboarding checklist."] if gems else [],
            caveats=["Stratified estimates skip strata with a constant control outcome and lean slightly upward."],
        )
    )

    champs = t.loc[t["quadrant"] == "Champion", "feature"].tolist()
    wide = t.loc[t["quadrant"] == "Widely used, low impact", "feature"].tolist()
    rising = t.loc[t["trend"] == "rising", "feature"].tolist()
    declining = t.loc[t["trend"] == "declining", "feature"].tolist()
    actions = []
    if champs:
        actions.append(f"Protect champions ({', '.join(champs)}): reliability first, no disruptive redesigns.")
    if wide:
        actions.append(f"Do not market {', '.join(wide)} as retention drivers; popularity is not impact.")
    if declining:
        actions.append(f"Investigate why the effect of {', '.join(declining)} is fading.")
    stages.append(
        StageSummary(
            stage="Active",
            headline=f"{len(champs)} champions carry retention; {len(wide)} popular features show no causal lift.",
            key_features=[
                FeatureCallout(
                    feature=r.feature,
                    effect_pp=round(r.psm_att * 100, 1),
                    evidence=f"adoption {r.adoption_rate:.0%}, PSM {_pp(r.psm_att)}, trend {r.trend}",
                )
                for r in t[t["quadrant"] == "Champion"].sort_values("psm_att", ascending=False).head(5).itertuples()
            ],
            actions=actions,
            caveats=[f"Trend labels use only the last 3 months (rising: {', '.join(rising) or 'none'})."],
        )
    )

    ren = t[t["confirmed"] & (t["psm_att"] > 0)].sort_values("psm_att", ascending=False).head(5)
    harmful = t.loc[t["harmful"], "feature"].tolist()
    fp = t.loc[t["naive_confirmed"] & ~t["confirmed"], "feature"].tolist()
    stages.append(
        StageSummary(
            stage="Renewal",
            headline=(
                f"After matching on merchant size and profile, {int((t['confirmed'] & (t['psm_att'] > 0)).sum())} "
                f"features causally lift renewal; {len(fp)} naive 'drivers' disappear."
            ),
            key_features=[
                FeatureCallout(
                    feature=r.feature,
                    effect_pp=round(r.psm_att * 100, 1),
                    evidence=(
                        f"PSM ATT {_pp(r.psm_att)} (95% CI {_pp(r.psm_ci_low)} to {_pp(r.psm_ci_high)}), "
                        f"significant in {r.months_significant}/6 months"
                    ),
                )
                for r in ren.itertuples()
            ],
            actions=(
                ([f"Review {', '.join(harmful)}: it is associated with LOWER renewal after matching."] if harmful else [])
                + ["Use renewal drivers in the account-health score and CSM playbooks."]
            ),
            caveats=[
                "Causal only under unconfoundedness: unobserved factors (e.g. owner engagement) can still bias results.",
                f"Naive comparisons would have wrongly credited: {', '.join(fp) or 'none'}.",
            ],
        )
    )
    return FeatureImpactSummary(
        stages=stages,
        data_note="Synthetic data - all numbers are generated by a simulator with a known ground truth.",
    )


# ----------------------------------------------------------------------------- LLM
_PROMPT = (
    "You are a senior data scientist writing for product managers of a POS SaaS product. "
    "Using ONLY the facts below (do not invent numbers or features), write one summary per "
    "lifecycle stage: Onboarding, Active, Renewal. effect_pp is in percentage points. "
    "Mention that the data is synthetic in data_note.\n\nFACTS:\n{facts}"
)


def _llm_client(provider: str):
    import instructor  # lazy

    if provider == "openai":
        from openai import OpenAI  # lazy

        return instructor.from_openai(OpenAI())
    if provider == "gemini":
        model = os.getenv("FEATURE_IMPACT_LLM_MODEL", DEFAULT_MODELS["gemini"])
        return instructor.from_provider(f"google/{model}")
    raise ValueError(f"unknown provider {provider!r}")


def llm_summary(table: pd.DataFrame, provider: str) -> FeatureImpactSummary:
    client = _llm_client(provider)
    prompt = _PROMPT.format(facts=json.dumps(build_facts(table), indent=1))
    kwargs = dict(response_model=FeatureImpactSummary, messages=[{"role": "user", "content": prompt}])
    if provider == "openai":
        kwargs["model"] = os.getenv("FEATURE_IMPACT_LLM_MODEL", DEFAULT_MODELS["openai"])
    return client.chat.completions.create(**kwargs)


def summarize(table: pd.DataFrame, provider: str | None = None) -> tuple[FeatureImpactSummary, str]:
    """Return (summary, source) where source is the provider used or ``template``."""
    provider = (provider or os.getenv("FEATURE_IMPACT_LLM", "none")).lower()
    if provider in ("", "none", "template"):
        return template_summary(table), "template"
    try:
        return llm_summary(table, provider), provider
    except Exception as exc:  # noqa: BLE001 - any provider failure falls back to the template
        warnings.warn(f"LLM summary via {provider!r} failed ({exc!r}); using template fallback.", stacklevel=2)
        return template_summary(table), "template"


def to_markdown(summary: FeatureImpactSummary) -> str:
    lines = []
    for s in summary.stages:
        lines += [f"## {s.stage}", "", f"**{s.headline}**", ""]
        if s.key_features:
            lines += ["| Feature | Effect | Evidence |", "|---|---:|---|"]
            lines += [f"| `{k.feature}` | {k.effect_pp:+.1f} pp | {k.evidence} |" for k in s.key_features]
            lines.append("")
        if s.actions:
            lines += ["Actions:", *[f"- {a}" for a in s.actions], ""]
        if s.caveats:
            lines += ["Caveats:", *[f"- {c}" for c in s.caveats], ""]
    lines += [f"_{summary.data_note}_", ""]
    return "\n".join(lines)
