"""
plot_spmotif_bias.py
====================
Generates three SPMotif prequential compression ratio plots:

    1. spmotif_bias_comparison.png  — two-panel (GIN backbone | GraphGPS backbone)
    2. spmotif_bias_combined.png    — single figure, all 8 model × backbone lines

Reads from:
    ../results/SPMotif_{0.5,0.7,0.9}/prequential_{model}_{backbone}.json

Saves to:
    ../results/spmotif_bias_comparison.png
    ../results/spmotif_bias_combined.png

Usage:
    python prequential/plot_spmotif_bias.py
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

# ── config ─────────────────────────────────────────────────────────────────────
BIASES  = ["0.5", "0.7", "0.9"]
COMBOS  = [
    ("GNN", "GIN_encoder"),
    ("GNN", "GraphGPS_encoder"),
    ("CAL", "GIN_encoder"),
    ("CAL", "GraphGPS_encoder"),
    ("CEL", "GIN_encoder"),
    ("CEL", "GraphGPS_encoder"),
    ("ICL", "GIN_encoder"),
    ("ICL", "GraphGPS_encoder"),
]

MODEL_COLORS    = {"GNN": "#4C72B0", "CAL": "#DD8452", "CEL": "#55A868", "ICL": "#C44E52"}
BACKBONE_LS     = {"GIN_encoder": "--", "GraphGPS_encoder": "-"}
BACKBONE_MARKER = {"GIN_encoder": "o",  "GraphGPS_encoder": "s"}
X               = np.array([0.5, 0.7, 0.9])
X_LABELS        = ["bias = 0.5\n(weak spurious)", "bias = 0.7\n(medium)", "bias = 0.9\n(strong spurious)"]


# ── data loading ───────────────────────────────────────────────────────────────

def load_results() -> dict:
    """Return results[(model, backbone)] = [ratio_0.5, ratio_0.7, ratio_0.9]."""
    results = {}
    for model, backbone in COMBOS:
        vals = []
        for b in BIASES:
            path = RESULTS_DIR / f"SPMotif_{b}" / f"prequential_{model}_{backbone}.json"
            with open(path) as f:
                d = json.load(f)
            vals.append(d["compression_ratio_mean"])
        results[(model, backbone)] = vals
    return results


# ── shared drawing helpers ─────────────────────────────────────────────────────

# Per-model annotation offsets (dx_pts, dy_pts) for the GPS panel where values
# are tightly clustered — stagger labels so they don't overlap.
_GPS_ANNOT_OFFSETS = {
    "GNN": ( 18,  4),
    "CAL": ( 18, -12),
    "CEL": (-18, -12),
    "ICL": (-18,  4),
}

def _draw_lines(ax, results, backbones, stagger_gps=False, annotate_offset_gin=-14):
    """Draw lines + markers for the given backbones onto ax."""
    for model, backbone in COMBOS:
        if backbone not in backbones:
            continue
        vals   = results[(model, backbone)]
        col    = MODEL_COLORS[model]
        ls     = BACKBONE_LS[backbone]
        mk     = BACKBONE_MARKER[backbone]
        is_gps = "GraphGPS" in backbone

        # line segments — dotted where ratio > 1.0
        for i in range(len(vals) - 1):
            seg_style = ":" if max(vals[i], vals[i + 1]) > 1.0 else ls
            ax.plot([X[i], X[i + 1]], [vals[i], vals[i + 1]],
                    color=col, linewidth=2.0, linestyle=seg_style)

        # markers (red outline = failure)
        for xi, yi in zip(X, vals):
            fail = yi > 1.0
            ax.plot(xi, yi, marker=mk, markersize=9, color=col, linestyle="None",
                    markeredgecolor="red" if fail else col,
                    markeredgewidth=2.5 if fail else 0, zorder=5)

        # value annotations
        for xi, yi in zip(X, vals):
            if is_gps and stagger_gps:
                dx, dy = _GPS_ANNOT_OFFSETS[model]
            else:
                dx, dy = (0, annotate_offset_gin)
            ax.annotate(f"{yi:.3f}", (xi, yi),
                        textcoords="offset points", xytext=(dx, dy),
                        ha="center", fontsize=7.5, color=col, fontweight="bold")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.3, alpha=0.6)
    ax.set_xticks(X)
    ax.set_xticklabels(X_LABELS, fontsize=9)
    ax.set_xlabel("Spurious Bias", fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Plot 1: two-panel (GIN | GraphGPS) ────────────────────────────────────────

def plot_two_panel(results: dict, out_path: Path):
    fig, (ax_gin, ax_gps) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    fig.suptitle(
        "SPMotif — Prequential Compression Ratio vs. Spurious Bias",
        fontsize=13, fontweight="bold",
    )

    panels = [
        (ax_gin, ["GIN_encoder"],      "GIN backbone",      False),
        (ax_gps, ["GraphGPS_encoder"], "GraphGPS backbone", True),
    ]

    for ax, backbones, title, is_gps_panel in panels:
        _draw_lines(ax, results, backbones, stagger_gps=is_gps_panel)
        all_vals = [v for m, b in COMBOS if b in backbones for v in results[(m, b)]]

        if is_gps_panel:
            ymin = 0.0
            ymax = 0.25
        else:
            ymax = max(all_vals) * 1.20 + 0.05
            ymin = max(0, min(all_vals) - 0.05)

        ax.set_ylim(ymin, ymax)
        ax.axhspan(1.0, ymax, alpha=0.07, color="red")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Compression Ratio", fontsize=10)

    # legend
    legend_elements = [
        plt.Line2D([0], [0], color=MODEL_COLORS["GNN"], lw=2.5, marker="o", label="GNN"),
        plt.Line2D([0], [0], color=MODEL_COLORS["CAL"], lw=2.5, marker="o", label="CAL"),
        plt.Line2D([0], [0], color=MODEL_COLORS["CEL"], lw=2.5, marker="o", label="CEL"),
        plt.Line2D([0], [0], color=MODEL_COLORS["ICL"], lw=2.5, marker="o", label="ICL"),
        plt.Line2D([0], [0], color="black",  lw=1.5, linestyle="--", label="Uniform baseline (1.0)"),
        plt.Line2D([0], [0], color="gray",   lw=2.0, linestyle=":",  label="Above 1.0 (no compression)"),
        plt.Line2D([0], [0], color="gray",   lw=0,   marker="o", markersize=9,
                   markeredgecolor="red", markeredgewidth=2.5, label="Failed point"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=7, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Plot 2: single combined figure (all 8 lines) ───────────────────────────────

def plot_combined(results: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(
        "SPMotif — Prequential Compression Ratio vs. Spurious Bias\n(all models × backbones)",
        fontsize=13, fontweight="bold",
    )

    all_vals = [v for vals in results.values() for v in vals]
    ymax = max(all_vals) * 1.20 + 0.05

    _draw_lines(ax, results, ["GIN_encoder", "GraphGPS_encoder"],
                stagger_gps=True, annotate_offset_gin=-14)

    # invisible lines for legend
    for model, backbone in COMBOS:
        col = MODEL_COLORS[model]
        ls  = BACKBONE_LS[backbone]
        mk  = BACKBONE_MARKER[backbone]
        bb_lbl = "GPS" if "GraphGPS" in backbone else "GIN"
        ax.plot([], [], color=col, linewidth=2.0, linestyle=ls,
                marker=mk, markersize=8, label=f"{model} + {bb_lbl}")

    ax.axhspan(1.0, ymax, alpha=0.07, color="red")
    ax.text(0.915, 1.03, "No compression\n(above baseline)", fontsize=8,
            color="red", alpha=0.7, va="bottom")
    ax.set_xlim(0.43, 0.97)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Compression Ratio", fontsize=11)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=9,
              ncol=2, title="Model + Backbone", title_fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = load_results()

    plot_two_panel(results, RESULTS_DIR / "spmotif_bias_comparison.png")
    plot_combined (results, RESULTS_DIR / "spmotif_bias_combined.png")
