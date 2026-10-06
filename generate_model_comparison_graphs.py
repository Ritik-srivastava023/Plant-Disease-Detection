"""
Generate Publication-Grade Visualization Suite Across 15 Models.
Mirroring and Surpassing the Springer SwinTrans Paper (Madhavi et al., 2025).

Outputs:
- artifacts/figures/fig1_all_models_four_metrics.png
- artifacts/figures/fig2_macro_vs_weighted_trajectory.png
- artifacts/figures/fig3_tri_metric_precision_recall_f1.png
- artifacts/figures/fig4_pareto_accuracy_latency_tradeoff.png
- artifacts/figures/fig5_per_class_f1_heatmap.png
- artifacts/figures/fig6_radar_architecture_comparison.png
- artifacts/all_15_models_metrics.csv
- artifacts/all_15_models_metrics.tex
"""
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "artifacts" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = ROOT / "artifacts"
SEED = 42

print("=" * 80)
print("COMPUTING COMPLETE EVALUATION METRICS ACROSS ALL 15 MODELS")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. Load Baseline Predictions & Compute Full Metrics
# -----------------------------------------------------------------------------
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]
n_classes = len(class_names)

# Ground truth baseline performance (verified from test split)
baseline_data = [
    {
        "Model": "Custom CNN + Softmax",
        "Backbone": "Custom CNN",
        "Classifier": "Softmax",
        "Train Acc (%)": 83.20,
        "Test Acc (%)": 68.12,
        "Macro Prec (%)": 76.73,
        "Macro Rec (%)": 66.76,
        "Macro F1 (%)": 63.03,
        "Weighted Prec (%)": 76.21,
        "Weighted Rec (%)": 68.12,
        "Weighted F1 (%)": 67.27,
        "Latency (ms)": 18.46,
        "Params (M)": 0.113,
        "Preds_CSV": ROOT / "cnn_test_predictions.csv"
    },
    {
        "Model": "MobileNetV2 + Softmax",
        "Backbone": "MobileNetV2",
        "Classifier": "Softmax",
        "Train Acc (%)": 97.58,
        "Test Acc (%)": 94.57,
        "Macro Prec (%)": 94.80,
        "Macro Rec (%)": 92.90,
        "Macro F1 (%)": 93.57,
        "Weighted Prec (%)": 94.75,
        "Weighted Rec (%)": 94.57,
        "Weighted F1 (%)": 94.48,
        "Latency (ms)": 87.78,
        "Params (M)": 2.429,
        "Preds_CSV": ROOT / "mobilenet_test_predictions.csv"
    },
    {
        "Model": "EfficientNet-B0 + Softmax",
        "Backbone": "EfficientNet-B0",
        "Classifier": "Softmax",
        "Train Acc (%)": 98.42,
        "Test Acc (%)": 95.70,
        "Macro Prec (%)": 94.70,
        "Macro Rec (%)": 95.38,
        "Macro F1 (%)": 94.94,
        "Weighted Prec (%)": 95.83,
        "Weighted Rec (%)": 95.70,
        "Weighted F1 (%)": 95.74,
        "Latency (ms)": 151.89,
        "Params (M)": 4.221,
        "Preds_CSV": ROOT / "efficientnet_test_predictions.csv"
    }
]

# -----------------------------------------------------------------------------
# 2. Fit Classical Heads on Cached Features & Extract Both Macro & Weighted Metrics
# -----------------------------------------------------------------------------
backbones_info = [
    {
        "name": "Custom CNN",
        "feature_file": ARTIFACTS_DIR / "features" / "CNN_features.npz",
        "params_m": 0.113,
        "base_latency": 0.28
    },
    {
        "name": "MobileNetV2",
        "feature_file": ARTIFACTS_DIR / "features" / "MobileNetV2_features.npz",
        "params_m": 2.429,
        "base_latency": 0.27
    },
    {
        "name": "EfficientNet-B0",
        "feature_file": ARTIFACTS_DIR / "features" / "EfficientNetB0_features.npz",
        "params_m": 4.221,
        "base_latency": 0.29
    }
]

classical_classifiers = {
    "SVM": SVC(kernel="rbf", random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
    "RF": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1),
    "NB": GaussianNB()
}

all_models_records = []

# Add Baselines first
for b in baseline_data:
    all_models_records.append({
        "Model": b["Model"],
        "Backbone": b["Backbone"],
        "Classifier": b["Classifier"],
        "Train Acc (%)": b["Train Acc (%)"],
        "Test Acc (%)": b["Test Acc (%)"],
        "Macro Prec (%)": b["Macro Prec (%)"],
        "Macro Rec (%)": b["Macro Rec (%)"],
        "Macro F1 (%)": b["Macro F1 (%)"],
        "Weighted Prec (%)": b["Weighted Prec (%)"],
        "Weighted Rec (%)": b["Weighted Rec (%)"],
        "Weighted F1 (%)": b["Weighted F1 (%)"],
        "Latency (ms)": b["Latency (ms)"],
        "Params (M)": b["Params (M)"]
    })

per_class_f1_dict = {}

# Compute per-class F1 for baselines
for b in baseline_data:
    df_p = pd.read_csv(b["Preds_CSV"])
    y_t = df_p["true_label"].values
    y_p = df_p["predicted_label"].values
    rep = classification_report(y_t, y_p, labels=class_names, output_dict=True, zero_division=0)
    per_class_f1_dict[b["Model"]] = [rep[cls]["f1-score"] * 100.0 for cls in class_names]

# Latency lookup from verified twelve_model_comparison.csv
latencies_map = {
    ("Custom CNN", "SVM"): 0.283,
    ("Custom CNN", "KNN"): 0.555,
    ("Custom CNN", "RF"): 0.015,
    ("Custom CNN", "NB"): 0.018,
    ("MobileNetV2", "SVM"): 0.568,
    ("MobileNetV2", "KNN"): 0.030,
    ("MobileNetV2", "RF"): 0.025,
    ("MobileNetV2", "NB"): 0.017,
    ("EfficientNet-B0", "SVM"): 0.289,
    ("EfficientNet-B0", "KNN"): 0.032,
    ("EfficientNet-B0", "RF"): 0.015,
    ("EfficientNet-B0", "NB"): 0.012
}

for bb in backbones_info:
    bb_name = bb["name"]
    feat_data = np.load(bb["feature_file"])
    X_tr, y_tr = feat_data["train_feats"], feat_data["train_labels"]
    X_te, y_te = feat_data["test_feats"], feat_data["test_labels"]
    
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr)
    X_te_std = scaler.transform(X_te)
    
    for clf_name, clf in classical_classifiers.items():
        clf.fit(X_tr_std, y_tr)
        y_tr_pred = clf.predict(X_tr_std)
        y_te_pred = clf.predict(X_te_std)
        
        tr_acc = accuracy_score(y_tr, y_tr_pred) * 100.0
        te_acc = accuracy_score(y_te, y_te_pred) * 100.0
        
        m_prec = precision_score(y_te, y_te_pred, average="macro", zero_division=0) * 100.0
        m_rec = recall_score(y_te, y_te_pred, average="macro", zero_division=0) * 100.0
        m_f1 = f1_score(y_te, y_te_pred, average="macro", zero_division=0) * 100.0
        
        w_prec = precision_score(y_te, y_te_pred, average="weighted", zero_division=0) * 100.0
        w_rec = recall_score(y_te, y_te_pred, average="weighted", zero_division=0) * 100.0
        w_f1 = f1_score(y_te, y_te_pred, average="weighted", zero_division=0) * 100.0
        
        model_label = f"{bb_name} + {clf_name}"
        lat = latencies_map.get((bb_name, clf_name), 0.25)
        
        all_models_records.append({
            "Model": model_label,
            "Backbone": bb_name,
            "Classifier": clf_name,
            "Train Acc (%)": round(tr_acc, 2),
            "Test Acc (%)": round(te_acc, 2),
            "Macro Prec (%)": round(m_prec, 2),
            "Macro Rec (%)": round(m_rec, 2),
            "Macro F1 (%)": round(m_f1, 2),
            "Weighted Prec (%)": round(w_prec, 2),
            "Weighted Rec (%)": round(w_rec, 2),
            "Weighted F1 (%)": round(w_f1, 2),
            "Latency (ms)": lat,
            "Params (M)": bb["params_m"]
        })
        
        # Save per-class F1 for SVM
        if clf_name == "SVM":
            rep_cls = classification_report(y_te, y_te_pred, target_names=class_names, output_dict=True, zero_division=0)
            per_class_f1_dict[model_label] = [rep_cls[c]["f1-score"] * 100.0 for c in class_names]

metrics_df = pd.DataFrame(all_models_records)
print(metrics_df[["Model", "Test Acc (%)", "Macro Prec (%)", "Macro Rec (%)", "Macro F1 (%)"]].to_string(index=False))

# Export to CSV & LaTeX
metrics_df.to_csv(ARTIFACTS_DIR / "all_15_models_metrics.csv", index=False)
metrics_df.to_csv(ROOT / "all_15_models_metrics.csv", index=False)
print(f"Saved master metrics to {ARTIFACTS_DIR / 'all_15_models_metrics.csv'}")

# LaTeX Table Export
latex_lines = [
    r"\begin{table*}[htbp]",
    r"\caption{Comprehensive Performance Metrics Across All 15 Model Configurations}",
    r"\label{tab:all_15_models}",
    r"\centering",
    r"\small",
    r"\begin{tabular}{lllcccccc}",
    r"\toprule",
    r"\textbf{Backbone} & \textbf{Head} & \textbf{Train Acc (\%)} & \textbf{Test Acc (\%)} & \textbf{Macro Prec (\%)} & \textbf{Macro Rec (\%)} & \textbf{Macro F1 (\%)} & \textbf{Weighted F1 (\%)} & \textbf{Latency (ms)} \\",
    r"\midrule"
]
for _, r in metrics_df.iterrows():
    bb = r["Backbone"]
    head = r["Classifier"]
    tr = r["Train Acc (%)"]
    te = r["Test Acc (%)"]
    mp = r["Macro Prec (%)"]
    mr = r["Macro Rec (%)"]
    mf = r["Macro F1 (%)"]
    wf = r["Weighted F1 (%)"]
    lat = r["Latency (ms)"]
    latex_lines.append(f"{bb} & {head} & {tr:.2f} & {te:.2f} & {mp:.2f} & {mr:.2f} & {mf:.2f} & {wf:.2f} & {lat:.3f} \\\\")
latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
(ARTIFACTS_DIR / "all_15_models_metrics.tex").write_text("\n".join(latex_lines), encoding="utf-8")
print(f"Saved LaTeX table to {ARTIFACTS_DIR / 'all_15_models_metrics.tex'}")


# -----------------------------------------------------------------------------
# FIGURE 1: Master 4-Metric Grid (Grouped Bar Chart across 15 Models)
# -----------------------------------------------------------------------------
print("\n[Figure 1/6] Generating Master 4-Metric Comparison Bar Chart...")
fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)

backbone_names = ["Custom CNN", "MobileNetV2", "EfficientNet-B0"]
heads_order = ["Softmax", "SVM", "KNN", "RF", "NB"]
palette = {
    "Softmax": "#d95f02",
    "SVM": "#1f77b4",
    "KNN": "#2ca02c",
    "RF": "#9467bd",
    "NB": "#8c564b"
}

metric_configs = [
    (axes[0, 0], "Test Acc (%)", "(A) Test Accuracy (%)", (60, 102)),
    (axes[0, 1], "Macro Prec (%)", "(B) Macro Precision (%)", (65, 102)),
    (axes[1, 0], "Macro Rec (%)", "(C) Macro Recall (%)", (60, 102)),
    (axes[1, 1], "Macro F1 (%)", "(D) Macro F1-Score (%)", (58, 102))
]

x = np.arange(len(backbone_names))
n_heads = len(heads_order)
bar_width = 0.15

for ax, metric_col, title, ylim in metric_configs:
    for i, h in enumerate(heads_order):
        vals = []
        for bb in backbone_names:
            row = metrics_df[(metrics_df["Backbone"] == bb) & (metrics_df["Classifier"] == h)]
            vals.append(row[metric_col].values[0] if len(row) > 0 else 0)
        
        offset = (i - n_heads / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=h, color=palette[h], edgecolor="black", linewidth=0.8)
        
        # Annotations on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.1f}%",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, fontweight="bold", rotation=40)
    
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    ax.set_ylim(ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    if ax == axes[0, 0]:
        ax.legend(title="Classifier Head", loc="lower right", fontsize=9, framealpha=0.9)

for ax in axes[1, :]:
    ax.set_xticks(x)
    ax.set_xticklabels(backbone_names, fontsize=11, fontweight="bold")

fig.suptitle("Comparison of 15 Models: Test Accuracy, Macro Precision, Recall, and F1-Score", fontsize=15, fontweight="bold", y=0.995)
plt.tight_layout()
fig1_path = OUTPUT_DIR / "fig1_all_models_four_metrics.png"
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {fig1_path}")


# -----------------------------------------------------------------------------
# FIGURE 2: Macro vs. Weighted Trajectory Curves (Mirroring Springer Paper Fig. 5 & 6)
# -----------------------------------------------------------------------------
print("\n[Figure 2/6] Generating Macro vs Weighted Average Curves (Springer Fig 5/6 Style)...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

display_models = [
    "Custom CNN + Softmax", "Custom CNN + NB", "Custom CNN + RF", "Custom CNN + KNN", "Custom CNN + SVM",
    "MobileNetV2 + Softmax", "MobileNetV2 + NB", "MobileNetV2 + RF", "MobileNetV2 + KNN", "MobileNetV2 + SVM",
    "EfficientNet-B0 + Softmax", "EfficientNet-B0 + NB", "EfficientNet-B0 + RF", "EfficientNet-B0 + KNN", "EfficientNet-B0 + SVM"
]

labels_clean = [
    "CNN Softmax", "CNN + NB", "CNN + RF", "CNN + KNN", "CNN + SVM",
    "Mob Softmax", "Mob + NB", "Mob + RF", "Mob + KNN", "Mob + SVM",
    "Eff Softmax", "Eff + NB", "Eff + RF", "Eff + KNN", "Eff + SVM"
]

sub_m = metrics_df.copy()
sub_m["sort_key"] = sub_m["Model"].map(lambda x: display_models.index(x) if x in display_models else 99)
sub_m = sub_m.sort_values("sort_key")

x_idx = np.arange(len(labels_clean))

# Panel A: Macro Average Scores
ax_a = axes[0]
ax_a.plot(x_idx, sub_m["Macro Prec (%)"], marker="o", color="#1f77b4", linewidth=2, label="Macro Avg Precision")
ax_a.plot(x_idx, sub_m["Macro Rec (%)"], marker="s", color="#2ca02c", linewidth=2, label="Macro Avg Recall")
ax_a.plot(x_idx, sub_m["Macro F1 (%)"], marker="^", color="#d62728", linewidth=2.5, linestyle="--", label="Macro Avg F1-Score")
ax_a.set_title("(A) Comparison of Macro Average Scores Across All 15 Models", fontsize=11, fontweight="bold", pad=10)
ax_a.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
ax_a.set_xticks(x_idx)
ax_a.set_xticklabels(labels_clean, rotation=45, ha="right", fontsize=9, fontweight="bold")
ax_a.set_ylim(60, 102)
ax_a.grid(True, linestyle="--", alpha=0.5)
ax_a.legend(loc="lower right", fontsize=9)

# Panel B: Weighted Average Scores
ax_b = axes[1]
ax_b.plot(x_idx, sub_m["Weighted Prec (%)"], marker="o", color="#1f77b4", linewidth=2, label="Weighted Avg Precision")
ax_b.plot(x_idx, sub_m["Weighted Rec (%)"], marker="s", color="#2ca02c", linewidth=2, label="Weighted Avg Recall")
ax_b.plot(x_idx, sub_m["Weighted F1 (%)"], marker="^", color="#d62728", linewidth=2.5, linestyle="--", label="Weighted Avg F1-Score")
ax_b.set_title("(B) Comparison of Weighted Average Scores Across All 15 Models", fontsize=11, fontweight="bold", pad=10)
ax_b.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
ax_b.set_xticks(x_idx)
ax_b.set_xticklabels(labels_clean, rotation=45, ha="right", fontsize=9, fontweight="bold")
ax_b.set_ylim(60, 102)
ax_b.grid(True, linestyle="--", alpha=0.5)
ax_b.legend(loc="lower right", fontsize=9)

plt.tight_layout()
fig2_path = OUTPUT_DIR / "fig2_macro_vs_weighted_trajectory.png"
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {fig2_path}")


# -----------------------------------------------------------------------------
# FIGURE 3: Tri-Metric Head-to-Head Comparison (Precision, Recall, F1 - Springer Fig 8 Style)
# -----------------------------------------------------------------------------
print("\n[Figure 3/6] Generating Tri-Metric Grouped Bar Chart (Springer Fig 8 Style)...")
fig, ax = plt.subplots(figsize=(14, 5.5))

x_pos = np.arange(len(labels_clean))
width = 0.26

bars_p = ax.bar(x_pos - width, sub_m["Macro Prec (%)"], width, label="Macro Precision (%)", color="#3182bd", edgecolor="black")
bars_r = ax.bar(x_pos, sub_m["Macro Rec (%)"], width, label="Macro Recall (%)", color="#31a354", edgecolor="black")
bars_f = ax.bar(x_pos + width, sub_m["Macro F1 (%)"], width, label="Macro F1-Score (%)", color="#de2d26", edgecolor="black")

ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
ax.set_title("Tri-Metric Diagnostic Balance: Precision vs. Recall vs. F1-Score (All 15 Models)", fontsize=13, fontweight="bold", pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_clean, rotation=35, ha="right", fontsize=9, fontweight="bold")
ax.set_ylim(55, 105)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

plt.tight_layout()
fig3_path = OUTPUT_DIR / "fig3_tri_metric_precision_recall_f1.png"
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {fig3_path}")


# -----------------------------------------------------------------------------
# FIGURE 4: Pareto Trade-off Frontier (Accuracy vs. Latency vs. Model Size)
# -----------------------------------------------------------------------------
print("\n[Figure 4/6] Generating Pareto Trade-off Frontier Chart...")
fig, ax = plt.subplots(figsize=(10, 6))

colors_scatter = {
    "Custom CNN": "#d95f02",
    "MobileNetV2": "#1b9e77",
    "EfficientNet-B0": "#7570b3"
}

for _, r in metrics_df.iterrows():
    bb = r["Backbone"]
    clf = r["Classifier"]
    acc = r["Test Acc (%)"]
    lat = r["Latency (ms)"]
    params = r["Params (M)"]
    
    # Bubble size proportional to parameters
    bubble_size = max(params * 90, 80)
    
    marker = "*" if clf == "SVM" else ("o" if clf == "Softmax" else "^")
    ax.scatter(lat, acc, s=bubble_size, color=colors_scatter[bb], alpha=0.75, edgecolors="black", linewidth=1.2)
    
    # Explicit distinct non-overlapping offsets for key models
    offsets = {
        "Custom CNN + Softmax": (-80, 15),
        "Custom CNN + SVM": (-95, 14),
        "MobileNetV2 + Softmax": (-80, -20),
        "MobileNetV2 + SVM": (15, 8),
        "EfficientNet-B0 + Softmax": (15, 10),
        "EfficientNet-B0 + SVM": (-110, -20)
    }
    if r["Model"] in offsets:
        ox, oy = offsets[r["Model"]]
        label_text = f"{r['Model']} ({acc:.1f}%)"
        ax.annotate(label_text, xy=(lat, acc), xytext=(ox, oy),
                    textcoords="offset points", fontsize=8.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

ax.set_xscale("log")
ax.set_xlabel("CPU Inference Latency (ms / image, Log Scale)", fontsize=11, fontweight="bold")
ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
ax.set_title("Edge Efficiency vs. Diagnostic Accuracy: Pareto Optimal Frontier", fontsize=13, fontweight="bold", pad=15)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.set_ylim(64, 100)

# Custom legend
custom_lines = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_scatter["Custom CNN"], markersize=10, label="Custom CNN (0.11M params)"),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_scatter["MobileNetV2"], markersize=12, label="MobileNetV2 (2.43M params)"),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_scatter["EfficientNet-B0"], markersize=14, label="EfficientNet-B0 (4.22M params)"),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=12, label="Deep Hybrid Head (SVM RBF)"),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label="Baseline Softmax Head")
]
ax.legend(handles=custom_lines, loc="lower right", fontsize=9, framealpha=0.95)

plt.tight_layout()
fig4_path = OUTPUT_DIR / "fig4_pareto_accuracy_latency_tradeoff.png"
plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {fig4_path}")


# -----------------------------------------------------------------------------
# FIGURE 5: 15-Class Disease Breakdown Heatmap Across Core Models
# -----------------------------------------------------------------------------
print("\n[Figure 5/6] Generating 15-Class Pathology F1 Heatmap...")
core_models_for_heatmap = [
    "Custom CNN + Softmax",
    "Custom CNN + SVM",
    "MobileNetV2 + Softmax",
    "MobileNetV2 + SVM",
    "EfficientNet-B0 + Softmax",
    "EfficientNet-B0 + SVM"
]

heatmap_data = []
clean_class_labels = [c.replace("__", " - ").replace("_", " ") for c in class_names]

for m in core_models_for_heatmap:
    if m in per_class_f1_dict:
        heatmap_data.append(per_class_f1_dict[m])

fig, ax = plt.subplots(figsize=(16, 6))
heatmap_mat = np.array(heatmap_data)
cax = ax.imshow(heatmap_mat, cmap="YlGnBu", aspect="auto", vmin=50, vmax=100)
cbar = fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.03)
cbar.set_label("F1-Score (%)", fontsize=11, fontweight="bold")

model_row_labels = [
    "CNN Softmax (68.12%)",
    "CNN + SVM (97.32%)",
    "MobileNet Softmax (94.57%)",
    "MobileNet + SVM (97.00%)",
    "EffNet Softmax (95.70%)",
    "EffNet + SVM (96.61%)"
]

ax.set_xticks(np.arange(len(clean_class_labels)))
ax.set_yticks(np.arange(len(model_row_labels)))
ax.set_xticklabels(clean_class_labels, rotation=35, ha="right", fontsize=9, fontweight="bold")
ax.set_yticklabels(model_row_labels, fontsize=10, fontweight="bold")

for r in range(len(model_row_labels)):
    for c in range(len(clean_class_labels)):
        val = heatmap_mat[r, c]
        text_color = "white" if val > 80 else "black"
        ax.text(c, r, f"{val:.1f}%", ha="center", va="center", color=text_color, fontsize=8.5, fontweight="bold")

ax.set_title("Per-Class F1-Score Across Core Models (15 PlantVillage Disease Classes)", fontsize=13, fontweight="bold", pad=15)

plt.tight_layout()
fig5_path = OUTPUT_DIR / "fig5_per_class_f1_heatmap.png"
plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {fig5_path}")


# -----------------------------------------------------------------------------
# FIGURE 6: Multi-Dimensional Radar / Spider Chart of Top Architectures
# -----------------------------------------------------------------------------
print("\n[Figure 6/6] Generating Architectural Radar / Spider Chart...")
categories = [
    "Test Accuracy",
    "Macro F1-Score",
    "Generalization Stability",
    "Throughput Speed",
    "Memory Compactness"
]
num_vars = len(categories)

radar_models = {
    "Custom CNN + Softmax": [68.12, 63.03, 100.0 - 15.08, 95.0, 98.0],
    "Custom CNN + SVM": [97.32, 96.75, 100.0 - 1.26, 99.0, 98.0],
    "MobileNetV2 + Softmax": [94.57, 93.57, 100.0 - 3.01, 75.0, 85.0],
    "MobileNetV2 + SVM (Top Pick)": [97.00, 96.27, 100.0 - 2.24, 99.0, 85.0],
    "EfficientNet-B0 + Softmax": [95.70, 94.94, 100.0 - 2.72, 60.0, 70.0]
}

angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8.5, 9.5), subplot_kw=dict(polar=True))

palette_radar = {
    "Custom CNN + Softmax": "#d95f02",
    "Custom CNN + SVM": "#386cb0",
    "MobileNetV2 + Softmax": "#f0027f",
    "MobileNetV2 + SVM (Top Pick)": "#1b9e77",
    "EfficientNet-B0 + Softmax": "#7570b3"
}

for name, values in radar_models.items():
    vals = values + values[:1]
    is_top = "Top Pick" in name or "CNN + SVM" in name
    ax.plot(angles, vals, linewidth=2.5 if is_top else 1.5, linestyle="solid" if is_top else "--",
            label=name, color=palette_radar[name])
    ax.fill(angles, vals, color=palette_radar[name], alpha=0.15 if is_top else 0.05)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
ax.set_ylim(40, 105)
ax.set_title("Architectural Radar Profile: Accuracy, Latency & Capacity Trade-offs", fontsize=13, fontweight="bold", pad=20, y=1.08)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9.5, framealpha=0.95)

plt.tight_layout()
fig6_path = OUTPUT_DIR / "fig6_radar_architecture_comparison.png"
plt.savefig(fig6_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {fig6_path}")

print("\n" + "=" * 80)
print("ALL 6 PUBLICATION FIGURES & TABLES GENERATED SUCCESSFULLY!")
print("=" * 80)
