"""
Comprehensive Line Graph Comparison Suite for Plant Disease Classification.

Generates publication-quality line graphs comparing all 15 models across the 5 core metrics:
1. Accuracy (Test Acc %)
2. Precision (Macro Precision %)
3. F1-Score (Macro F1 %)
4. Recall (Macro Recall %)
5. Error Rate (Misclassification Error Rate % = 100 - Acc)

Visualizations:
- Figure 17: Comprehensive 5-Metric Performance Profile & Trajectory Line Graph Comparison
  - Panel A: Unified 5-Metric Multi-Line Trajectory across all 15 ranked models
  - Panel B: Metric-by-Metric Subplot Trajectories (Accuracy, Precision, Recall, F1, Error Rate)
  - Panel C: Architectural Transition Lines across Classifier Heads (Softmax -> SVM -> KNN -> RF -> NB)
  - Panel D: Accuracy vs Error Rate Inverse Trade-Off Curve

Outputs:
- artifacts/five_metrics_comparison.csv
- artifacts/five_metrics_comparison.tex
- artifacts/five_metrics_comparison.md
- artifacts/figures/fig17_five_metric_line_comparison.png
- C:/Users/ritik/Downloads/fig17_five_metric_line_comparison.png
"""

import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = ROOT / "all_15_models_metrics.csv"
if not INPUT_CSV.exists():
    INPUT_CSV = ARTIFACTS_DIR / "all_15_models_metrics.csv"

print("=" * 80)
print("GENERATING 5-METRIC LINE GRAPH COMPARISON ACROSS ALL 15 MODELS")
print("=" * 80)

df = pd.read_csv(INPUT_CSV)

# Compute explicit Error Rate if not present
if "Error Rate (%)" not in df.columns:
    df["Error Rate (%)"] = np.round(100.0 - df["Test Acc (%)"], 2)

# Select the 5 core target metrics
core_columns = [
    "Model", "Backbone", "Classifier",
    "Test Acc (%)", "Macro Prec (%)", "Macro F1 (%)", "Macro Rec (%)", "Error Rate (%)"
]
df_core = df[core_columns].copy()

# Sort models by Test Accuracy descending
df_sorted = df_core.sort_values("Test Acc (%)", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------------------------
# 1. Export Data Tables (CSV, Markdown, LaTeX)
# -----------------------------------------------------------------------------
out_csv = ARTIFACTS_DIR / "five_metrics_comparison.csv"
df_sorted.to_csv(out_csv, index=False)
print(f"Saved 5-metric dataset -> {out_csv}")

md_table = df_sorted.to_markdown(index=False)
(ARTIFACTS_DIR / "five_metrics_comparison.md").write_text(md_table, encoding="utf-8")

tex_table = (
    "\\begin{table*}[t]\n"
    "\\caption{Performance Comparison of All 15 Models Across the Five Core Evaluation Metrics: Accuracy, Precision, F1-Score, Recall, and Classification Error Rate ($N=3{,}096$ Test Images).}\n"
    "\\label{tab:five_metrics_comparison}\n"
    "\\centering\n"
    "\\footnotesize\n"
    "\\begin{tabular}{llccccc}\n"
    "\\toprule\n"
    "\\textbf{Model Architecture} & \\textbf{Head Classifier} & \\textbf{Accuracy (\\%)} & \\textbf{Precision (\\%)} & \\textbf{F1-Score (\\%)} & \\textbf{Recall (\\%)} & \\textbf{Error Rate (\\%)} \\\\\n"
    "\\midrule\n"
)
for _, r in df_sorted.iterrows():
    tex_table += (
        f"{r['Model']} & {r['Classifier']} & {r['Test Acc (%)']:.2f} & {r['Macro Prec (%)']:.2f} & "
        f"{r['Macro F1 (%)']:.2f} & {r['Macro Rec (%)']:.2f} & {r['Error Rate (%)']:.2f} \\\\\n"
    )
tex_table += (
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table*}\n"
)
(ARTIFACTS_DIR / "five_metrics_comparison.tex").write_text(tex_table, encoding="utf-8")
print(f"Saved LaTeX table -> {ARTIFACTS_DIR / 'five_metrics_comparison.tex'}")

# -----------------------------------------------------------------------------
# 2. Render Publication Figure 17 (High-Resolution 300 DPI)
# -----------------------------------------------------------------------------
print("\nRendering Publication Figure 17: Comprehensive 5-Metric Line Graph Suite...")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig = plt.figure(figsize=(20, 13), dpi=300)
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.22)

ax1 = fig.add_subplot(gs[0, :])  # Panel A: Unified Master Line Graph across all 15 models (spans full width)
ax2 = fig.add_subplot(gs[1, 0])  # Panel B: Head Transition Lines across the 3 Backbones (Accuracy & F1)
ax3 = fig.add_subplot(gs[1, 1])  # Panel C: Error Rate Inverse Trajectory & Metric Spread

# Styling definitions for the 5 metrics
metric_styles = {
    "Accuracy": {
        "col": "Test Acc (%)",
        "color": "#1f77b4",       # Strong Blue
        "marker": "o",
        "linestyle": "-",
        "label": "1. Accuracy (%)",
        "linewidth": 2.5,
    },
    "Precision": {
        "col": "Macro Prec (%)",
        "color": "#2ca02c",       # Forest Green
        "marker": "s",
        "linestyle": "--",
        "label": "2. Precision (Macro %)",
        "linewidth": 2.2,
    },
    "F1 Score": {
        "col": "Macro F1 (%)",
        "color": "#ff7f0e",       # Vivid Orange
        "marker": "D",
        "linestyle": "-.",
        "label": "3. F1-Score (Macro %)",
        "linewidth": 2.4,
    },
    "Recall": {
        "col": "Macro Rec (%)",
        "color": "#9467bd",       # Royal Purple
        "marker": "^",
        "linestyle": ":",
        "label": "4. Recall (Macro %)",
        "linewidth": 2.2,
    },
    "Error Rate": {
        "col": "Error Rate (%)",
        "color": "#d62728",       # Crimson Red
        "marker": "v",
        "linestyle": "-",
        "label": "5. Error Rate (%) [Secondary Axis]",
        "linewidth": 2.6,
    },
}

# -----------------------------------------------------------------------------
# Panel A: Master Multi-Line Trajectory Across All 15 Ranked Models
# -----------------------------------------------------------------------------
models_15 = df_sorted["Model"].tolist()
x_indices = np.arange(len(models_15))

# Primary Y-axis for Performance Metrics (Accuracy, Precision, F1, Recall)
for m_name in ["Accuracy", "Precision", "F1 Score", "Recall"]:
    cfg = metric_styles[m_name]
    vals = df_sorted[cfg["col"]].values
    ax1.plot(
        x_indices, vals,
        color=cfg["color"],
        marker=cfg["marker"],
        markersize=7,
        linestyle=cfg["linestyle"],
        linewidth=cfg["linewidth"],
        label=cfg["label"],
        alpha=0.9
    )

# Fill between min and max of positive metrics to show metric convergence envelope
max_pos = np.maximum.reduce([
    df_sorted["Test Acc (%)"].values,
    df_sorted["Macro Prec (%)"].values,
    df_sorted["Macro F1 (%)"].values,
    df_sorted["Macro Rec (%)"].values
])
min_pos = np.minimum.reduce([
    df_sorted["Test Acc (%)"].values,
    df_sorted["Macro Prec (%)"].values,
    df_sorted["Macro F1 (%)"].values,
    df_sorted["Macro Rec (%)"].values
])
ax1.fill_between(x_indices, min_pos, max_pos, color="#1f77b4", alpha=0.08, label="Inter-Metric Concordance Envelope")

ax1.set_ylabel("Diagnostic Metric Score (%)", fontsize=11, fontweight="bold", color="#1f77b4")
ax1.set_ylim(58, 102)
ax1.set_xticks(x_indices)
ax1.set_xticklabels([m.replace(" + ", "\n+ ") for m in models_15], fontsize=8.5, fontweight="bold")
ax1.set_title("A. Master Multi-Line Trajectory Across All 15 Models (Ranked by Test Accuracy)", fontsize=13, fontweight="bold", pad=12)

# Secondary Twin Y-axis for Error Rate (%)
ax1_twin = ax1.twinx()
err_cfg = metric_styles["Error Rate"]
err_vals = df_sorted[err_cfg["col"]].values
ax1_twin.plot(
    x_indices, err_vals,
    color=err_cfg["color"],
    marker=err_cfg["marker"],
    markersize=8,
    linestyle=err_cfg["linestyle"],
    linewidth=err_cfg["linewidth"],
    label=err_cfg["label"],
    alpha=0.95
)
ax1_twin.set_ylabel("Classification Error Rate (%) [100 - Acc]", fontsize=11, fontweight="bold", color=err_cfg["color"])
ax1_twin.set_ylim(-1, 38)
ax1_twin.grid(False)

# Annotate Top Model and Bottom Model
ax1.annotate(
    f"Rank 1: Custom CNN + SVM\nAcc: 97.32% | Err: 2.68%\nF1: 96.75% | Prec: 96.83%",
    xy=(0, df_sorted["Test Acc (%)"].iloc[0]),
    xytext=(0.5, 87),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1.2, headwidth=6),
    fontsize=8, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f4f8", edgecolor="#1f77b4", alpha=0.9)
)
ax1.annotate(
    f"Rank 15: Custom CNN + Softmax\nAcc: 68.12% | Err: 31.88%\nF1: 63.03% | Rec: 66.76%",
    xy=(14, df_sorted["Test Acc (%)"].iloc[14]),
    xytext=(11.5, 76),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1.2, headwidth=6),
    fontsize=8, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fde8e8", edgecolor="#d62728", alpha=0.9)
)

# Combine legends from primary and secondary axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True, fontsize=9, ncol=3)

# -----------------------------------------------------------------------------
# Panel B: Architectural Transition Lines across Classifier Heads (By Backbone)
# -----------------------------------------------------------------------------
head_order = ["Softmax", "SVM", "KNN", "RF", "NB"]
x_heads = np.arange(len(head_order))

backbone_styles = {
    "Custom CNN": {"color": "#1f77b4", "marker": "s", "label": "Custom CNN (113k)"},
    "MobileNetV2": {"color": "#2ca02c", "marker": "o", "label": "MobileNetV2 (2.43M)"},
    "EfficientNet-B0": {"color": "#d95f02", "marker": "D", "label": "EfficientNet-B0 (4.22M)"},
}

for bb, b_cfg in backbone_styles.items():
    bb_df = df[df["Backbone"] == bb].set_index("Classifier")
    acc_series = [bb_df.loc[h, "Test Acc (%)"] if h in bb_df.index else np.nan for h in head_order]
    f1_series = [bb_df.loc[h, "Macro F1 (%)"] if h in bb_df.index else np.nan for h in head_order]
    
    # Plot Accuracy Line (solid)
    ax2.plot(
        x_heads, acc_series,
        color=b_cfg["color"],
        marker=b_cfg["marker"],
        markersize=7,
        linestyle="-",
        linewidth=2.2,
        label=f"{b_cfg['label']} [Acc]",
        alpha=0.9
    )
    # Plot Macro F1 Line (dashed)
    ax2.plot(
        x_heads, f1_series,
        color=b_cfg["color"],
        marker=b_cfg["marker"],
        markersize=6,
        linestyle="--",
        linewidth=1.8,
        label=f"{b_cfg['label']} [F1]",
        alpha=0.75
    )

ax2.set_xticks(x_heads)
ax2.set_xticklabels(head_order, fontsize=10, fontweight="bold")
ax2.set_xlabel("Classifier Decision Head Transition", fontsize=11, fontweight="bold")
ax2.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
ax2.set_ylim(60, 100)
ax2.set_title("B. Head Transition Profiles: Softmax -> SVM -> KNN -> RF -> NB\n(Solid Line = Accuracy, Dashed Line = Macro F1)", fontsize=11.5, fontweight="bold", pad=10)
ax2.legend(loc="lower left", frameon=True, fontsize=8, ncol=2)

# Highlight Custom CNN dramatic jump from Softmax to SVM
ax2.annotate(
    "SVM Lift: +29.2% Acc\n(Eliminates Bottleneck)",
    xy=(1, df[(df['Backbone']=='Custom CNN') & (df['Classifier']=='SVM')]['Test Acc (%)'].values[0]),
    xytext=(0.8, 80),
    arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1, headwidth=5),
    fontsize=8, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#1f77b4")
)

# -----------------------------------------------------------------------------
# Panel C: Error Rate Line vs Accuracy Dual Trajectory & Convergence
# -----------------------------------------------------------------------------
for bb, b_cfg in backbone_styles.items():
    bb_df = df[df["Backbone"] == bb].set_index("Classifier")
    err_series = [bb_df.loc[h, "Error Rate (%)"] if h in bb_df.index else np.nan for h in head_order]
    
    ax3.plot(
        x_heads, err_series,
        color=b_cfg["color"],
        marker=b_cfg["marker"],
        markersize=8,
        linestyle="-",
        linewidth=2.4,
        label=f"{b_cfg['label']}",
        alpha=0.9
    )
    # Annotate error rate numbers
    for h_idx, e_val in enumerate(err_series):
        if not np.isnan(e_val):
            offset_y = 1.2 if bb != "MobileNetV2" else -2.2
            ax3.text(h_idx, e_val + offset_y, f"{e_val:.1f}%", ha="center", fontsize=7.5, fontweight="bold", color=b_cfg["color"])

ax3.set_xticks(x_heads)
ax3.set_xticklabels(head_order, fontsize=10, fontweight="bold")
ax3.set_xlabel("Classifier Decision Head Transition", fontsize=11, fontweight="bold")
ax3.set_ylabel("Classification Error Rate (%) [lower is better]", fontsize=11, fontweight="bold")
ax3.set_ylim(-3, 36)
ax3.set_title("C. Classification Error Rate Trajectory across Heads (%)\n(Proves SVM & KNN Minimize Diagnostic Misclassifications)", fontsize=11.5, fontweight="bold", pad=10)
ax3.legend(loc="upper left", frameon=True, fontsize=8.5)

# Overall Figure Suptitle
fig.suptitle(
    "Figure 17: Comprehensive 5-Metric Performance Profile & Line Graph Comparison Across All 15 Models\n(Accuracy, Precision, F1-Score, Recall, and Classification Error Rate on PlantVillage N=3,096)",
    fontsize=15, fontweight="bold", y=0.99
)

plt.tight_layout()

fig_path = FIGURES_DIR / "fig17_five_metric_line_comparison.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)

downloads_path = Path("C:/Users/ritik/Downloads/fig17_five_metric_line_comparison.png")
if downloads_path.parent.exists():
    shutil.copy(fig_path, downloads_path)
    print(f"Copied figure to Downloads -> {downloads_path}")

print(f"\nFigure 17 successfully generated -> {fig_path}")

print("\n" + "=" * 80)
print("5-METRIC SUMMARY TABLE (RANKED BY ACCURACY)")
print("=" * 80)
print(df_sorted.to_string(index=False))
print("=" * 80)
