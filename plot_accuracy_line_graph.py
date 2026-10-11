"""
Dedicated Accuracy-Only Line Graph Comparison Suite for All 15 Models.

Generates a publication-grade, dedicated line graph focusing solely on:
Metric 1: Classification Test Accuracy (%) Across All 15 Models.

Provides two clear visual perspectives:
1. Master Accuracy Trajectory (Ranked Line across all 15 distinct configurations)
   - Labeled data points for every single model
   - Shaded SOTA threshold bands (>=95% and >=90%)
   - Backbone-specific color-coded markers
2. Backbone-Wise Head Transition Lines (Custom CNN vs MobileNetV2 vs EfficientNet-B0)
   - X-axis: Softmax -> SVM -> KNN -> RF -> NB
   - 3 comparative lines demonstrating architectural behavior

Outputs:
- artifacts/figures/accuracy_line_graph_all_15_models.png
- C:/Users/ritik/Downloads/accuracy_line_graph_all_15_models.png
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

CSV_PATH = ROOT / "all_15_models_metrics.csv"
if not CSV_PATH.exists():
    CSV_PATH = ARTIFACTS_DIR / "all_15_models_metrics.csv"

df = pd.read_csv(CSV_PATH)
df_sorted = df.sort_values("Test Acc (%)", ascending=False).reset_index(drop=True)

print("=" * 80)
print("GENERATING DEDICATED ACCURACY-ONLY LINE GRAPH FOR ALL 15 MODELS")
print("=" * 80)

# Backbone color palette
backbone_palette = {
    "Custom CNN": "#1f77b4",       # Steel Blue
    "MobileNetV2": "#2ca02c",      # Emerald Green
    "EfficientNet-B0": "#d95f02",  # Vivid Orange
}

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), dpi=300, gridspec_kw={'height_ratios': [1.2, 1.0]})

# =============================================================================
# View 1: Master Accuracy Line Trajectory Across All 15 Models (Ranked 1 to 15)
# =============================================================================
models_15 = df_sorted["Model"].tolist()
acc_15 = df_sorted["Test Acc (%)"].values
x_15 = np.arange(len(models_15))

# Shaded background zones
ax1.axhspan(95.0, 100.0, color="#2ecc71", alpha=0.08, label="High-Performance SOTA Tier (>= 95%)")
ax1.axhspan(90.0, 95.0, color="#f1c40f", alpha=0.08, label="Standard Edge Deployment Tier (90% - 95%)")
ax1.axhline(95.0, color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.7)
ax1.axhline(90.0, color="#f39c12", linestyle="--", linewidth=1.2, alpha=0.7)

# Main connecting line
ax1.plot(x_15, acc_15, color="#2c3e50", linestyle="-", linewidth=2.6, zorder=3, alpha=0.85, label="Test Accuracy Trajectory")

# Plot colored scatter nodes based on backbone
for bb_name, bb_col in backbone_palette.items():
    bb_indices = [i for i, b in enumerate(df_sorted["Backbone"]) if b == bb_name]
    ax1.scatter(
        [x_15[i] for i in bb_indices],
        [acc_15[i] for i in bb_indices],
        color=bb_col,
        s=110,
        zorder=5,
        edgecolor="black",
        linewidth=1.2,
        label=f"{bb_name} Configuration"
    )

# Value annotations above each point
for i, (m, a) in enumerate(zip(models_15, acc_15)):
    va_dir = "bottom" if i < 13 else "top"
    offset_y = 0.9 if i < 13 else -2.2
    font_col = "#2c3e50" if a >= 90 else "#c0392b"
    ax1.annotate(
        f"{a:.2f}%",
        xy=(i, a),
        xytext=(i, a + offset_y),
        ha="center",
        va=va_dir,
        fontsize=8.5,
        fontweight="bold",
        color=font_col,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.85)
    )

ax1.set_xticks(x_15)
ax1.set_xticklabels([m.replace(" + ", "\n+ ") for m in models_15], fontsize=8.5, fontweight="bold")
ax1.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
ax1.set_ylim(60, 102)
ax1.set_title("A. Master Test Accuracy Line Trajectory (All 15 Models Ranked from Highest to Lowest)", fontsize=13, fontweight="bold", pad=12)
ax1.legend(loc="lower left", frameon=True, fontsize=9.5, ncol=3)

# Highlight Peak Model
peak_idx = 0
ax1.annotate(
    f"Best Model: Custom CNN + SVM (97.32%)\nParameter Footprint: 0.113M (Lowest)\nInference Latency: 0.28ms",
    xy=(0, acc_15[0]),
    xytext=(1.2, 91.5),
    arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=6),
    fontsize=9, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ebf5fb", edgecolor="#1f77b4", linewidth=1.2)
)

# =============================================================================
# View 2: Backbone-Wise Head Transition Lines Across the 5 Classifier Heads
# =============================================================================
head_order = ["Softmax", "SVM", "KNN", "RF", "NB"]
x_heads = np.arange(len(head_order))

for bb_name, bb_col in backbone_palette.items():
    bb_df = df[df["Backbone"] == bb_name].set_index("Classifier")
    acc_series = [bb_df.loc[h, "Test Acc (%)"] if h in bb_df.index else np.nan for h in head_order]
    
    # Plot line for this backbone
    line = ax2.plot(
        x_heads, acc_series,
        color=bb_col,
        marker="o",
        markersize=9,
        linewidth=2.8,
        linestyle="-",
        label=f"{bb_name}",
        alpha=0.9
    )
    
    # Annotate points on the lines
    for h_idx, val in enumerate(acc_series):
        if not np.isnan(val):
            # Dynamic text placement to avoid overlap
            offset_y = 1.3
            if bb_name == "MobileNetV2" and h_idx == 1:
                offset_y = -2.2
            elif bb_name == "EfficientNet-B0" and h_idx in [1, 2]:
                offset_y = -2.2
            elif bb_name == "Custom CNN" and h_idx == 0:
                offset_y = 1.4
            
            ax2.text(
                h_idx, val + offset_y,
                f"{val:.2f}%",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color=bb_col,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor=bb_col, alpha=0.75, linewidth=0.8)
            )

ax2.set_xticks(x_heads)
ax2.set_xticklabels(head_order, fontsize=11, fontweight="bold")
ax2.set_xlabel("Classifier Head Transition (End-to-End Softmax vs. Shallow Feature Heads)", fontsize=11.5, fontweight="bold")
ax2.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
ax2.set_ylim(62, 102)
ax2.set_title("B. Architectural Head Transitions by Backbone (Comparing Decision Head Impact on Accuracy)", fontsize=13, fontweight="bold", pad=12)
ax2.legend(loc="lower left", frameon=True, fontsize=10)

# Annotate Custom CNN Lift
ax2.annotate(
    "Custom CNN Jump: +29.20% Accuracy\n(SVM completely resolves capacity bottleneck)",
    xy=(1, df[(df['Backbone']=='Custom CNN') & (df['Classifier']=='SVM')]['Test Acc (%)'].values[0]),
    xytext=(0.75, 78),
    arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.4, headwidth=6),
    fontsize=9, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#1f77b4", linewidth=1.2)
)

fig.suptitle(
    "Figure 18: Pure Accuracy Line Graph Comparison Across All 15 Models\n(PlantVillage Benchmark, Untouched N=3,096 Test Split, Seed 42)",
    fontsize=15, fontweight="bold", y=0.99
)

plt.tight_layout()

# Save paths
fig_path = FIGURES_DIR / "accuracy_line_graph_all_15_models.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)

downloads_path = Path("C:/Users/ritik/Downloads/accuracy_line_graph_all_15_models.png")
shutil.copy(fig_path, downloads_path)

print(f"\nFigure saved successfully -> {fig_path}")
print(f"Copied directly to Downloads -> {downloads_path}")

# Print summary table of accuracy only
acc_summary = df_sorted[["Model", "Backbone", "Classifier", "Test Acc (%)"]].copy()
acc_summary["Rank"] = range(1, len(acc_summary) + 1)
print("\n" + "=" * 80)
print("ALL 15 MODELS ACCURACY RANKING TABLE")
print("=" * 80)
print(acc_summary[["Rank", "Model", "Backbone", "Classifier", "Test Acc (%)"]].to_string(index=False))
print("=" * 80)

