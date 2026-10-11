"""
Dedicated Line Graph Generation Suite for All 5 Core Evaluation Metrics:
1. Accuracy (%)
2. Precision (Macro %)
3. F1-Score (Macro %)
4. Recall (Macro %)
5. Error Rate (Classification Misclassification %)

Generates standalone, publication-quality 300 DPI line graphs for EACH metric individually,
with complete data point annotations, benchmark tier shading, and backbone head transitions.
Also generates a 5-panel master composite grid.

Outputs:
- artifacts/figures/accuracy_line_graph_all_15_models.png
- artifacts/figures/precision_line_graph_all_15_models.png
- artifacts/figures/f1_score_line_graph_all_15_models.png
- artifacts/figures/recall_line_graph_all_15_models.png
- artifacts/figures/error_rate_line_graph_all_15_models.png
- artifacts/figures/all_5_individual_line_graphs_grid.png
All copied to C:/Users/ritik/Downloads/
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
DOWNLOADS_DIR = Path("C:/Users/ritik/Downloads")

CSV_PATH = ROOT / "all_15_models_metrics.csv"
if not CSV_PATH.exists():
    CSV_PATH = ARTIFACTS_DIR / "all_15_models_metrics.csv"

df = pd.read_csv(CSV_PATH)
if "Error Rate (%)" not in df.columns:
    df["Error Rate (%)"] = np.round(100.0 - df["Test Acc (%)"], 2)

print("=" * 80)
print("GENERATING DEDICATED INDIVIDUAL LINE GRAPHS FOR ALL 5 METRICS")
print("=" * 80)

backbone_palette = {
    "Custom CNN": "#1f77b4",       # Steel Blue
    "MobileNetV2": "#2ca02c",      # Emerald Green
    "EfficientNet-B0": "#d95f02",  # Vivid Orange
}

head_order = ["Softmax", "SVM", "KNN", "RF", "NB"]
x_heads = np.arange(len(head_order))

# Definitions for each of the 5 metrics
metrics_config = [
    {
        "id": "accuracy",
        "name": "Accuracy",
        "col": "Test Acc (%)",
        "title": "Classification Test Accuracy (%)",
        "y_label": "Test Accuracy (%)",
        "y_limits": (60, 102),
        "is_higher_better": True,
        "line_color": "#1f77b4",
        "file_name": "accuracy_line_graph_all_15_models.png",
        "figure_num": "Figure 18A",
    },
    {
        "id": "precision",
        "name": "Precision",
        "col": "Macro Prec (%)",
        "title": "Macro-Averaged Precision (%)",
        "y_label": "Macro Precision (%)",
        "y_limits": (65, 102),
        "is_higher_better": True,
        "line_color": "#2ca02c",
        "file_name": "precision_line_graph_all_15_models.png",
        "figure_num": "Figure 18B",
    },
    {
        "id": "f1_score",
        "name": "F1-Score",
        "col": "Macro F1 (%)",
        "title": "Macro-Averaged F1-Score (%)",
        "y_label": "Macro F1-Score (%)",
        "y_limits": (58, 102),
        "is_higher_better": True,
        "line_color": "#ff7f0e",
        "file_name": "f1_score_line_graph_all_15_models.png",
        "figure_num": "Figure 18C",
    },
    {
        "id": "recall",
        "name": "Recall",
        "col": "Macro Rec (%)",
        "title": "Macro-Averaged Recall / Sensitivity (%)",
        "y_label": "Macro Recall (%)",
        "y_limits": (62, 102),
        "is_higher_better": True,
        "line_color": "#9467bd",
        "file_name": "recall_line_graph_all_15_models.png",
        "figure_num": "Figure 18D",
    },
    {
        "id": "error_rate",
        "name": "Error Rate",
        "col": "Error Rate (%)",
        "title": "Classification Error Rate (%) [Misclassification %]",
        "y_label": "Error Rate (%) [lower is better]",
        "y_limits": (-2, 36),
        "is_higher_better": False,
        "line_color": "#d62728",
        "file_name": "error_rate_line_graph_all_15_models.png",
        "figure_num": "Figure 18E",
    },
]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# =============================================================================
# Generate Each Individual 2-Panel Figure
# =============================================================================
for m_cfg in metrics_config:
    col = m_cfg["col"]
    m_name = m_cfg["name"]
    is_high = m_cfg["is_higher_better"]
    
    # Sort models by this specific metric (descending if higher better, ascending if lower better)
    df_m = df.sort_values(col, ascending=not is_high).reset_index(drop=True)
    models_list = df_m["Model"].tolist()
    vals = df_m[col].values
    x_idx = np.arange(len(models_list))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), dpi=300, gridspec_kw={'height_ratios': [1.2, 1.0]})
    
    # -------------------------------------------------------------------------
    # Panel 1: Master Ranked Trajectory
    # -------------------------------------------------------------------------
    if is_high:
        ax1.axhspan(95.0, 100.0, color="#2ecc71", alpha=0.08, label="High-Performance Tier (>= 95%)")
        ax1.axhspan(90.0, 95.0, color="#f1c40f", alpha=0.08, label="Standard Deployment Tier (90% - 95%)")
        ax1.axhline(95.0, color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.7)
        ax1.axhline(90.0, color="#f39c12", linestyle="--", linewidth=1.2, alpha=0.7)
    else:
        ax1.axhspan(0.0, 5.0, color="#2ecc71", alpha=0.08, label="Minimal Error Tier (<= 5%)")
        ax1.axhspan(5.0, 10.0, color="#f1c40f", alpha=0.08, label="Acceptable Error Tier (5% - 10%)")
        ax1.axhline(5.0, color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.7)
        ax1.axhline(10.0, color="#f39c12", linestyle="--", linewidth=1.2, alpha=0.7)
    
    # Main Line
    ax1.plot(x_idx, vals, color="#2c3e50", linestyle="-", linewidth=2.6, zorder=3, alpha=0.85, label=f"{m_name} Line")
    
    # Scatter points colored by backbone
    for bb_name, bb_col in backbone_palette.items():
        bb_indices = [i for i, b in enumerate(df_m["Backbone"]) if b == bb_name]
        ax1.scatter(
            [x_idx[i] for i in bb_indices],
            [vals[i] for i in bb_indices],
            color=bb_col,
            s=110,
            zorder=5,
            edgecolor="black",
            linewidth=1.2,
            label=f"{bb_name} Configuration"
        )
    
    # Numeric Annotations
    for i, (m, v) in enumerate(zip(models_list, vals)):
        if is_high:
            va_dir = "bottom" if i < 13 else "top"
            offset_y = 0.9 if i < 13 else -2.2
            font_col = "#2c3e50" if v >= 90 else "#c0392b"
        else:
            va_dir = "bottom" if i < 13 else "bottom"
            offset_y = 0.8
            font_col = "#27ae60" if v <= 5.0 else ("#f39c12" if v <= 10.0 else "#c0392b")
            
        ax1.annotate(
            f"{v:.2f}%",
            xy=(i, v),
            xytext=(i, v + offset_y),
            ha="center",
            va=va_dir,
            fontsize=8.5,
            fontweight="bold",
            color=font_col,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.85)
        )
        
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([m.replace(" + ", "\n+ ") for m in models_list], fontsize=8.5, fontweight="bold")
    ax1.set_ylabel(m_cfg["y_label"], fontsize=12, fontweight="bold")
    ax1.set_ylim(m_cfg["y_limits"])
    direction_txt = "Highest to Lowest" if is_high else "Lowest to Highest (Best to Worst)"
    ax1.set_title(f"A. Master {m_name} Line Trajectory (All 15 Models Ranked from {direction_txt})", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="lower left" if is_high else "upper left", frameon=True, fontsize=9.5, ncol=3)
    
    # Highlight Best Model
    best_m = models_list[0]
    best_v = vals[0]
    arrow_y = best_v - 7 if is_high else best_v + 5
    ax1.annotate(
        f"Optimal: {best_m} ({best_v:.2f}%)\n{m_name} Leader",
        xy=(0, best_v),
        xytext=(1.2, arrow_y),
        arrowprops=dict(facecolor=m_cfg["line_color"], shrink=0.08, width=1.4, headwidth=6),
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor=m_cfg["line_color"], linewidth=1.2)
    )

    # -------------------------------------------------------------------------
    # Panel 2: Backbone Transition Across Decision Heads
    # -------------------------------------------------------------------------
    for bb_name, bb_col in backbone_palette.items():
        bb_df = df[df["Backbone"] == bb_name].set_index("Classifier")
        series = [bb_df.loc[h, col] if h in bb_df.index else np.nan for h in head_order]
        
        ax2.plot(
            x_heads, series,
            color=bb_col,
            marker="o",
            markersize=9,
            linewidth=2.8,
            linestyle="-",
            label=f"{bb_name}",
            alpha=0.9
        )
        
        for h_idx, val in enumerate(series):
            if not np.isnan(val):
                offset_y = 1.3 if is_high else 1.2
                if bb_name == "MobileNetV2" and h_idx == 1:
                    offset_y = -2.2 if is_high else -2.0
                elif bb_name == "EfficientNet-B0" and h_idx in [1, 2]:
                    offset_y = -2.2 if is_high else -2.0
                    
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
    ax2.set_ylabel(m_cfg["y_label"], fontsize=12, fontweight="bold")
    ax2.set_ylim(m_cfg["y_limits"])
    ax2.set_title(f"B. {m_name} Architectural Transition Curves (Comparing Decision Head Impact by Backbone)", fontsize=13, fontweight="bold", pad=12)
    ax2.legend(loc="lower left" if is_high else "upper left", frameon=True, fontsize=10)

    fig.suptitle(
        f"{m_cfg['figure_num']}: Dedicated {m_cfg['title']} Line Graph Comparison Across All 15 Models\n(PlantVillage Benchmark, Untouched N=3,096 Test Split, Seed 42)",
        fontsize=15, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    
    save_path = FIGURES_DIR / m_cfg["file_name"]
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Copy to downloads and brain
    shutil.copy(save_path, DOWNLOADS_DIR / m_cfg["file_name"])
    brain_dir = Path("C:/Users/ritik/.gemini/antigravity/brain/7755c27b-df01-4dd3-978b-4b85b77c5270")
    if brain_dir.exists():
        shutil.copy(save_path, brain_dir / m_cfg["file_name"])
    print(f"Generated and copied: {m_cfg['file_name']}")

# =============================================================================
# Generate 5-Panel Master Composite Grid
# =============================================================================
print("\nGenerating 5-Panel Master Composite Grid...")
fig_comp, axes = plt.subplots(5, 1, figsize=(18, 22), dpi=300, sharex=True)

models_master = df.sort_values("Test Acc (%)", ascending=False)["Model"].tolist()
x_master = np.arange(len(models_master))

for idx, m_cfg in enumerate(metrics_config):
    ax = axes[idx]
    col = m_cfg["col"]
    m_name = m_cfg["name"]
    is_high = m_cfg["is_higher_better"]
    color = m_cfg["line_color"]
    
    vals_master = [df[df["Model"] == m][col].values[0] for m in models_master]
    
    ax.plot(x_master, vals_master, color=color, linestyle="-", linewidth=2.4, marker="o", markersize=6, alpha=0.9, label=m_name)
    ax.fill_between(x_master, vals_master, [m_cfg["y_limits"][0]] * len(x_master) if is_high else [m_cfg["y_limits"][1]] * len(x_master), color=color, alpha=0.08)
    
    for x_i, v in enumerate(vals_master):
        offset = 1.2 if is_high else 1.0
        ax.text(x_i, v + offset, f"{v:.1f}", ha="center", fontsize=7.5, fontweight="bold", color=color)
        
    ax.set_ylabel(f"{m_name} (%)", fontsize=10.5, fontweight="bold", color=color)
    ax.set_ylim(m_cfg["y_limits"])
    ax.set_title(f"{idx+1}. {m_cfg['title']}", fontsize=11.5, fontweight="bold", pad=6, loc="left", color=color)

axes[-1].set_xticks(x_master)
axes[-1].set_xticklabels([m.replace(" + ", "\n+ ") for m in models_master], fontsize=9, fontweight="bold")
axes[-1].set_xlabel("Model Architecture Configurations (Ranked 1 to 15 by Test Accuracy)", fontsize=11, fontweight="bold")

fig_comp.suptitle(
    "Figure 18: Complete 5-Metric Independent Line Graph Suite Across All 15 Models\n(Accuracy, Precision, F1-Score, Recall, and Classification Error Rate)",
    fontsize=16, fontweight="bold", y=0.995
)

plt.tight_layout()
comp_path = FIGURES_DIR / "all_5_individual_line_graphs_grid.png"
fig_comp.savefig(comp_path, dpi=300, bbox_inches="tight")
plt.close(fig_comp)

shutil.copy(comp_path, DOWNLOADS_DIR / "all_5_individual_line_graphs_grid.png")
brain_dir = Path("C:/Users/ritik/.gemini/antigravity/brain/7755c27b-df01-4dd3-978b-4b85b77c5270")
if brain_dir.exists():
    shutil.copy(comp_path, brain_dir / "all_5_individual_line_graphs_grid.png")
print("Generated and copied: all_5_individual_line_graphs_grid.png")

print("\n" + "=" * 80)
print("ALL 5 DEDICATED METRIC LINE GRAPHS SUCCESSFULLY CREATED AND EXPORTED")
print("=" * 80)
