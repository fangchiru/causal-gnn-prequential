"""
print_results.py
================
Reads prequential encoding results from ../results/ and prints a summary table
showing compression ratio (mean ± std) per Dataset × Model × Backbone.

A compression ratio < 1.0 means the model learned genuine structure
(it compressed the labels better than a uniform code).

Results are saved to:
    ../results/prequential_summary.csv
    ../results/prequential_compression_ratio.png
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

# ── Configuration ─────────────────────────────────────────────────────────────
DATASETS  = ["MUTAG", "IMDB-MULTI", "SPMotif_0.5", "SPMotif_0.7", "SPMotif_0.9"]
MODELS    = ["GNN", "CAL", "ICL"]
BACKBONES = ["GIN_encoder", "GraphGPS_encoder"]

BACKBONE_LABELS = {"GIN_encoder": "GIN", "GraphGPS_encoder": "GraphGPS"}
COLORS          = {"GIN_encoder": "steelblue", "GraphGPS_encoder": "coral"}

results_root = "results"
output_dir   = "results"
os.makedirs(output_dir, exist_ok=True)


# ── 1. Collect all results ────────────────────────────────────────────────────
# all_results[dataset][model][backbone] = {"ratio": (mean, std), "bits": (mean, std)} or None
all_results = {}

for dataset in DATASETS:
    all_results[dataset] = {}
    for model in MODELS:
        all_results[dataset][model] = {}
        for backbone in BACKBONES:
            file_path = os.path.join(
                results_root, dataset,
                f"prequential_{model}_{backbone}.json"
            )
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                all_results[dataset][model][backbone] = {
                    "ratio": (data["compression_ratio_mean"], data["compression_ratio_std"]),
                    "bits":  (data["total_bits_mean"],        data["total_bits_std"]),
                    "uniform_bits": data.get("uniform_bits", None),
                }
            else:
                all_results[dataset][model][backbone] = None


# ── 2. Print & save summary table ────────────────────────────────────────────
col_w = [14, 8, 12, 28]
header = (f"{'Dataset':<{col_w[0]}} | {'Model':<{col_w[1]}} | "
          f"{'Backbone':<{col_w[2]}} | {'Compression Ratio (mean ± std)':<{col_w[3]}}")
divider = "=" * len(header)

print(f"\n{divider}")
print(header)
print(f"  (ratio < 1.0 = model learned signal; ratio = 1.0 = no better than random)")
print(divider)

csv_rows = [["Dataset", "Model", "Backbone",
             "Compression Ratio mean", "Compression Ratio std",
             "Total Bits mean", "Total Bits std"]]

for dataset in DATASETS:
    # Find best (lowest) compression ratio for this dataset
    best_ratio = min(
        (all_results[dataset][m][b]["ratio"][0]
         for m in MODELS for b in BACKBONES
         if all_results[dataset][m][b]),
        default=None
    )

    for model in MODELS:
        for backbone in BACKBONES:
            r = all_results[dataset][model][backbone]
            label = BACKBONE_LABELS[backbone]
            ds_label = dataset if model == MODELS[0] and backbone == BACKBONES[0] else ""
            if r is None:
                print(f"{ds_label:<{col_w[0]}} | {model:<{col_w[1]}} | "
                      f"{label:<{col_w[2]}} | {'NO DATA':<{col_w[3]}}")
                csv_rows.append([dataset, model, label, "", "", "", ""])
            else:
                ratio_str = f"{r['ratio'][0]:.4f} ± {r['ratio'][1]:.4f}"
                mark = " ◀ best" if best_ratio and round(r["ratio"][0], 4) == round(best_ratio, 4) else ""
                print(f"{ds_label:<{col_w[0]}} | {model:<{col_w[1]}} | "
                      f"{label:<{col_w[2]}} | {ratio_str}{mark}")
                csv_rows.append([
                    dataset, model, label,
                    f"{r['ratio'][0]:.4f}", f"{r['ratio'][1]:.4f}",
                    f"{r['bits'][0]:.1f}",  f"{r['bits'][1]:.1f}",
                ])
    print("-" * len(header))

# Save CSV
csv_path = os.path.join(output_dir, "prequential_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)
print(f"\nSummary table saved to: {csv_path}")


# ── 3. Bar chart — compression ratio, one panel per dataset ──────────────────
n_datasets = len(DATASETS)
fig, axes = plt.subplots(1, n_datasets, figsize=(4 * n_datasets, 5), sharey=True)
if n_datasets == 1:
    axes = [axes]

fig.suptitle("Prequential Encoding — Compression Ratio per Dataset\n"
             "(< 1.0 = model learns signal; lower is better)", fontsize=13, y=1.02)

x      = np.arange(len(MODELS))
width  = 0.35

for col, dataset in enumerate(DATASETS):
    ax = axes[col]

    for i, backbone in enumerate(BACKBONES):
        means, stds = [], []
        for model in MODELS:
            r = all_results[dataset][model][backbone]
            if r:
                means.append(r["ratio"][0])
                stds.append(r["ratio"][1])
            else:
                means.append(0)
                stds.append(0)

        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width,
               yerr=stds, capsize=5,
               label=BACKBONE_LABELS[backbone],
               color=COLORS[backbone],
               alpha=0.85)

    # Uniform baseline reference line at 1.0
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.2, label="Uniform (1.0)")

    ax.set_title(dataset, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylim(0, 1.15)
    if col == 0:
        ax.set_ylabel("Compression Ratio", fontsize=10)
    if col == 0:
        ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plot_path = os.path.join(output_dir, "prequential_compression_ratio.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Bar chart saved to: {plot_path}")
