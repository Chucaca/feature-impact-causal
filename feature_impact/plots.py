"""Static charts for the README (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .quadrants import CHAMPION, HIDDEN_GEM, LOW_LOW, WIDE_LOW  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"  # PSM / after
ORANGE = "#eb6834"  # naive / before
AQUA = "#1baf7a"
GRAY = "#9a9893"
QUAD_COLORS = {CHAMPION: BLUE, HIDDEN_GEM: AQUA, WIDE_LOW: ORANGE, LOW_LOW: GRAY}


def _style(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold", pad=26 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")


def hero_chart(table: pd.DataFrame, path: Path) -> None:
    """Naive diff vs PSM estimate (95% CI) vs ground truth, per feature (renewal)."""
    t = table.sort_values("true_att").reset_index(drop=True)
    y = range(len(t))
    fig, ax = plt.subplots(figsize=(9, 8.5), facecolor=SURFACE)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.hlines(y, t["psm_ci_low"] * 100, t["psm_ci_high"] * 100, color=BLUE, linewidth=2, alpha=0.55)
    ax.scatter(t["naive_diff"] * 100, y, s=60, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="Naive: users minus non-users")
    ax.scatter(t["psm_att"] * 100, y, s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4,
               label="PSM estimate (95% CI)")
    ax.scatter(t["true_att"] * 100, y, s=110, marker="|", color=INK, linewidth=2.2, zorder=5,
               label="Ground truth (simulator)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(t["feature"], fontsize=9, color=INK)
    ax.set_xlabel("Effect on renewal rate (percentage points, mean of 6 monthly cohorts)", color=INK_2, fontsize=9.5)
    _style(ax, "Naive comparisons credit almost every feature; PSM recovers the truth",
           "Synthetic data: 60k merchant-months, 20 features, known causal effects")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def love_plot(balance: pd.DataFrame, feature: str, path: Path, threshold: float = 0.1) -> None:
    """|SMD| per covariate before vs after matching."""
    b = balance.assign(before=balance["smd_before"].abs(), after=balance["smd_after"].abs())
    b = b.sort_values("before").reset_index(drop=True)
    y = range(len(b))
    fig, ax = plt.subplots(figsize=(8, 7.5), facecolor=SURFACE)
    ax.axvline(threshold, color=INK_2, linestyle="--", linewidth=1)
    ax.text(threshold, len(b) - 0.3, f" |SMD| = {threshold}", color=INK_2, fontsize=8.5, va="bottom")
    ax.hlines(y, b["after"], b["before"], color=GRID, linewidth=1.5)
    ax.scatter(b["before"], y, s=55, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Before matching")
    ax.scatter(b["after"], y, s=55, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4, label="After matching")
    ax.set_yticks(list(y))
    ax.set_yticklabels(b["covariate"], fontsize=8.5, color=INK)
    ax.set_xlabel("Absolute standardized mean difference", color=INK_2, fontsize=9.5)
    n_bad_before = int((b["before"] > threshold).sum())
    n_bad_after = int((b["after"] > threshold).sum())
    _style(ax, f"Covariate balance for '{feature}'",
           f"Synthetic data, snapshot 1. Imbalanced covariates: {n_bad_before} before -> {n_bad_after} after")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def quadrant_chart(table: pd.DataFrame, path: Path) -> None:
    """Adoption vs causal impact bubble chart (bubble size = months significant)."""
    t = table
    a_thr = t.attrs.get("adoption_threshold", float(t["adoption_rate"].median()))
    i_thr = t.attrs.get("impact_threshold", 0.02)
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)
    ax.axvline(a_thr * 100, color=INK_2, linestyle="--", linewidth=1)
    ax.axhline(i_thr * 100, color=INK_2, linestyle="--", linewidth=1)
    ax.axhline(0, color=GRID, linewidth=1)
    for quad, color in QUAD_COLORS.items():
        q = t[t["quadrant"] == quad]
        if q.empty:
            continue
        ax.scatter(q["adoption_rate"] * 100, q["psm_att"] * 100, s=40 + 45 * q["months_significant"],
                   color=color, alpha=0.85, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([], [], s=80, color=color, label=quad)  # fixed-size legend handle
    _label_points(ax, fig, t)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel("Adoption: share of renewing merchants using the feature in T-4..T-2 (%)", color=INK_2, fontsize=9.5)
    ax.set_ylabel("Causal effect on renewal, PSM ATT (pp)", color=INK_2, fontsize=9.5)
    _style(ax, "Where to invest: adoption vs causal impact",
           "Synthetic data. Bubble size = months significant (of 6). High impact requires >=4/6 months.")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _label_points(ax, fig, t: pd.DataFrame) -> None:
    """Direct labels with a tiny greedy collision avoidance (display-space boxes)."""
    fig.canvas.draw()
    to_px = ax.transData.transform
    placed: list[tuple[float, float, float, float]] = []
    offsets = [(7, 4), (7, -12), (-7, 4), (-7, -12), (7, 14), (-7, 14), (7, -22), (-7, -22)]
    scale = fig.dpi / 72.0
    for r in t.sort_values("adoption_rate").itertuples():
        text = r.feature + (" (harmful)" if getattr(r, "harmful", False) else "")
        x, y = to_px((r.adoption_rate * 100, r.psm_att * 100))
        w, h = 4.9 * len(text) * scale, 10 * scale
        for dx, dy in offsets:
            x0 = x + dx * scale - (w if dx < 0 else 0)
            y0 = y + dy * scale
            box = (x0, y0, x0 + w, y0 + h)
            if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in placed):
                break
        placed.append(box)
        ax.annotate(text, (r.adoption_rate * 100, r.psm_att * 100), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, color=INK_2, ha="right" if dx < 0 else "left")
