"""
Comprehensive MRE (Magnitude of Relative Error) & MMRE (Mean Magnitude of Relative Error)
Benchmark and Visualization Suite for All 15 Plant Disease Classification Models.

Models Evaluated:
- 3 Backbones: Custom CNN (113k), MobileNetV2 (2.43M), EfficientNet-B0 (4.22M)
- 5 Decision Heads: Softmax, SVM (RBF), KNN (k=5), Random Forest (200 trees), Gaussian NB
Total: 15 Architectures on the Fixed Untouched Test Split (N = 3,096 images, Seed 42).

Metrics Computed:
- MMRE (Probabilistic Mean Magnitude of Relative Error): mean(1 - p_true)
- SD_MRE: Sample standard deviation of MRE across test samples
- MdMRE: Median Magnitude of Relative Error
- PRED(0.25): Percentage of test predictions with MRE <= 0.25 (confidence >= 75%)
- PRED(0.10): Percentage of test predictions with MRE <= 0.10 (confidence >= 90%)
- Class MMRE (%): Mean relative error in predicted class prevalence vs actual support
- Test Accuracy (%) & Error Rate (%)

Outputs:
- artifacts/mre_mmre_benchmark.csv
- artifacts/mre_mmre_benchmark.tex
- artifacts/mre_mmre_benchmark.md
- artifacts/figures/fig16_mre_mmre_all_15_models.png
- C:/Users/ritik/Downloads/fig16_mre_mmre_all_15_models.png
"""

import os
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Suppress logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
FEATURES_DIR = ARTIFACTS_DIR / "features"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

print("=" * 80)
print("MRE & MMRE BENCHMARK SUITE ACROSS ALL 15 MODELS")
print("=" * 80)

# Container for all results
all_models_data = []
sample_mre_dict = {}

# -----------------------------------------------------------------------------
# 1. Evaluate 3 Deep End-to-End Softmax Baselines (from exact test prediction CSVs)
# -----------------------------------------------------------------------------
print("\n[1/3] Loading Softmax baseline test predictions...")
softmax_files = [
    ("Custom CNN + Softmax", "Custom CNN", "Softmax", ROOT / "cnn_test_predictions.csv"),
    ("MobileNetV2 + Softmax", "MobileNetV2", "Softmax", ROOT / "mobilenet_test_predictions.csv"),
    ("EfficientNet-B0 + Softmax", "EfficientNet-B0", "Softmax", ROOT / "efficientnet_test_predictions.csv"),
]

for m_name, bb, head, f_path in softmax_files:
    if not f_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {f_path}")
    
    df = pd.read_csv(f_path)
    classes = [c.replace("prob_", "") for c in df.columns if c.startswith("prob_")]
    t_labels = df["true_label"].values
    p_labels = df["predicted_label"].values
    n_test = len(df)
    
    # Instance level MRE on true class probability target y = 1.0
    true_probs = np.array([df.loc[i, "prob_" + t_labels[i]] for i in range(n_test)], dtype=np.float64)
    mre_samples = np.clip(1.0 - true_probs, 0.0, 1.0)
    sample_mre_dict[m_name] = mre_samples
    
    mmre = float(np.mean(mre_samples))
    sd_mre = float(np.std(mre_samples, ddof=1))
    mdmre = float(np.median(mre_samples))
    pred25 = float(np.mean(mre_samples <= 0.25) * 100.0)
    pred10 = float(np.mean(mre_samples <= 0.10) * 100.0)
    
    # Class level prevalence MRE
    act_counts = df["true_label"].value_counts()
    pred_counts = df["predicted_label"].value_counts()
    class_mres = [abs(act_counts.get(c, 0) - pred_counts.get(c, 0)) / act_counts.get(c, 0) * 100.0 for c in classes]
    class_mmre = float(np.mean(class_mres))
    
    acc = float(np.mean(t_labels == p_labels) * 100.0)
    err_rate = 100.0 - acc
    
    all_models_data.append({
        "Model": m_name,
        "Backbone": bb,
        "Head": head,
        "Test Acc (%)": round(acc, 2),
        "Error Rate (%)": round(err_rate, 2),
        "MMRE": round(mmre, 4),
        "SD_MRE": round(sd_mre, 4),
        "MdMRE": round(mdmre, 4),
        "PRED(0.25) (%)": round(pred25, 2),
        "PRED(0.10) (%)": round(pred10, 2),
        "Class MMRE (%)": round(class_mmre, 2),
    })
    print(f"  {m_name:26s} | Acc: {acc:5.2f}% | MMRE: {mmre:.4f} | MdMRE: {mdmre:.4f} | PRED(0.25): {pred25:5.2f}% | Class MMRE: {class_mmre:5.2f}%")

# -----------------------------------------------------------------------------
# 2. Evaluate 12 Deep Hybrid Models (Penultimate Features + Classical Heads)
# -----------------------------------------------------------------------------
print("\n[2/3] Evaluating 12 Deep Penultimate Hybrids with Probabilistic Heads...")

backbones = [
    ("Custom CNN", "CNN_features.npz"),
    ("MobileNetV2", "MobileNetV2_features.npz"),
    ("EfficientNet-B0", "EfficientNetB0_features.npz"),
]

classifiers = {
    "SVM": lambda: SVC(kernel="rbf", probability=True, random_state=SEED),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "RF": lambda: RandomForestClassifier(n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1),
    "NB": lambda: GaussianNB(),
}

for bb_name, npz_name in backbones:
    npz_path = FEATURES_DIR / npz_name
    d = np.load(npz_path)
    X_tr, y_tr = d["train_feats"], d["train_labels"]
    X_te, y_te = d["test_feats"], d["test_labels"]
    n_test = len(y_te)
    u_classes = np.unique(y_te)
    
    # Feature standardization
    scaler = StandardScaler().fit(X_tr)
    X_tr_std = scaler.transform(X_tr)
    X_te_std = scaler.transform(X_te)
    
    for c_name, c_factory in classifiers.items():
        m_name = f"{bb_name} + {c_name}"
        t0 = time.perf_counter()
        clf = c_factory().fit(X_tr_std, y_tr)
        t_fit = time.perf_counter() - t0
        
        preds = clf.predict(X_te_std)
        probs = clf.predict_proba(X_te_std)
        
        acc = float(np.mean(preds == y_te) * 100.0)
        err_rate = 100.0 - acc
        
        # Instance level MRE on true class probability
        true_probs = probs[np.arange(n_test), y_te]
        mre_samples = np.clip(1.0 - true_probs, 0.0, 1.0)
        sample_mre_dict[m_name] = mre_samples
        
        mmre = float(np.mean(mre_samples))
        sd_mre = float(np.std(mre_samples, ddof=1))
        mdmre = float(np.median(mre_samples))
        pred25 = float(np.mean(mre_samples <= 0.25) * 100.0)
        pred10 = float(np.mean(mre_samples <= 0.10) * 100.0)
        
        # Class level MRE
        act_counts = pd.Series(y_te).value_counts()
        pred_counts = pd.Series(preds).value_counts()
        class_mres = [abs(act_counts.get(c, 0) - pred_counts.get(c, 0)) / act_counts.get(c, 0) * 100.0 for c in u_classes]
        class_mmre = float(np.mean(class_mres))
        
        all_models_data.append({
            "Model": m_name,
            "Backbone": bb_name,
            "Head": c_name,
            "Test Acc (%)": round(acc, 2),
            "Error Rate (%)": round(err_rate, 2),
            "MMRE": round(mmre, 4),
            "SD_MRE": round(sd_mre, 4),
            "MdMRE": round(mdmre, 4),
            "PRED(0.25) (%)": round(pred25, 2),
            "PRED(0.10) (%)": round(pred10, 2),
            "Class MMRE (%)": round(class_mmre, 2),
        })
        print(f"  {m_name:26s} | Acc: {acc:5.2f}% | MMRE: {mmre:.4f} | MdMRE: {mdmre:.4f} | PRED(0.25): {pred25:5.2f}% | Class MMRE: {class_mmre:5.2f}%")

df_results = pd.DataFrame(all_models_data).sort_values("Test Acc (%)", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------------------------
# 3. Export CSV, Markdown & LaTeX Tables
# -----------------------------------------------------------------------------
out_csv = ARTIFACTS_DIR / "mre_mmre_benchmark.csv"
df_results.to_csv(out_csv, index=False)
print(f"\nSaved CSV -> {out_csv}")

md_table = df_results.to_markdown(index=False)
(ARTIFACTS_DIR / "mre_mmre_benchmark.md").write_text(md_table, encoding="utf-8")

tex_table = (
    "\\begin{table*}[t]\n"
    "\\caption{Comprehensive Magnitude of Relative Error (MRE) and Mean MRE (MMRE) Benchmark Across All 15 Plant Disease Classification Architectures ($N=3{,}096$ Test Images).}\n"
    "\\label{tab:mre_mmre_benchmark}\n"
    "\\centering\n"
    "\\footnotesize\n"
    "\\begin{tabular}{llccccccc}\n"
    "\\toprule\n"
    "\\textbf{Model Architecture} & \\textbf{Classifier Head} & \\textbf{Acc (\\%)} & \\textbf{Err (\\%)} & \\textbf{MMRE} & \\textbf{SD\\textsubscript{MRE}} & \\textbf{MdMRE} & \\textbf{PRED(0.25) (\\%)} & \\textbf{Class MMRE (\\%)} \\\\\n"
    "\\midrule\n"
)
for _, r in df_results.iterrows():
    tex_table += (
        f"{r['Model']} & {r['Head']} & {r['Test Acc (%)']:.2f} & {r['Error Rate (%)']:.2f} & "
        f"{r['MMRE']:.4f} & {r['SD_MRE']:.4f} & {r['MdMRE']:.4f} & {r['PRED(0.25) (%)']:.2f} & {r['Class MMRE (%)']:.2f} \\\\\n"
    )
tex_table += (
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table*}\n"
)
(ARTIFACTS_DIR / "mre_mmre_benchmark.tex").write_text(tex_table, encoding="utf-8")
print(f"Saved LaTeX -> {ARTIFACTS_DIR / 'mre_mmre_benchmark.tex'}")

# -----------------------------------------------------------------------------
# 4. Generate Publication-Grade Figure 16 (4 Panels)
# -----------------------------------------------------------------------------
print("\n[3/3] Generating Figure 16: Publication-Grade Visual MRE/MMRE Benchmark...")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig = plt.figure(figsize=(18, 12), dpi=300)
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Color coding by backbone
backbone_colors = {
    "Custom CNN": "#1f77b4",       # Steel Blue
    "MobileNetV2": "#2ca02c",      # Emerald Green
    "EfficientNet-B0": "#d95f02",  # Vivid Orange
}

# Sort models by Test Accuracy descending for visual consistency
df_plot = df_results.sort_values("Test Acc (%)", ascending=True).reset_index(drop=True)
model_names = df_plot["Model"].tolist()
y_pos = np.arange(len(model_names))
bar_colors = [backbone_colors.get(b, "#333333") for b in df_plot["Backbone"]]

# -----------------------------------------------------------------------------
# Panel A: Probabilistic MMRE with SD_MRE Error Bars
# -----------------------------------------------------------------------------
mmre_vals = df_plot["MMRE"].values
sd_vals = df_plot["SD_MRE"].values

bars_a = ax1.barh(y_pos, mmre_vals, xerr=sd_vals, capsize=4, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(model_names, fontsize=8.5, fontweight="bold")
ax1.set_xlabel("Mean Magnitude of Relative Error (MMRE) [lower is better]", fontsize=10.5, fontweight="bold")
ax1.set_title("A. Probabilistic MMRE with Sample Standard Deviation Error Bars", fontsize=11.5, fontweight="bold", pad=10)

# Annotate values
for i, v in enumerate(mmre_vals):
    ax1.text(v + sd_vals[i] + 0.012, i, f"{v:.4f}", va="center", fontsize=8, fontweight="bold")
ax1.set_xlim(0, max(mmre_vals + sd_vals) * 1.15)

# -----------------------------------------------------------------------------
# Panel B: PRED(0.25) High-Confidence Precision Benchmark
# -----------------------------------------------------------------------------
pred25_vals = df_plot["PRED(0.25) (%)"].values
bars_b = ax2.barh(y_pos, pred25_vals, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.8)
ax2.axvline(90.0, color="#e74c3c", linestyle="--", linewidth=1.2, label="High-Reliability Threshold (90%)")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(model_names, fontsize=8.5, fontweight="bold")
ax2.set_xlabel("PRED(0.25) Metric (% of samples with MRE <= 0.25) [higher is better]", fontsize=10.5, fontweight="bold")
ax2.set_title("B. High-Confidence Precision: PRED(0.25) (Samples with Conf >= 75%)", fontsize=11.5, fontweight="bold", pad=10)

for i, v in enumerate(pred25_vals):
    ax2.text(v - 4.5, i, f"{v:.1f}%", va="center", ha="right", fontsize=8, fontweight="bold", color="white")
ax2.set_xlim(50, 100)
ax2.legend(loc="lower right", frameon=True)

# -----------------------------------------------------------------------------
# Panel C: MRE Boxplot / Distribution Across All 15 Models
# -----------------------------------------------------------------------------
# Select top 7 models and key baselines for clear boxplot visualization
box_models = [
    "Custom CNN + SVM", "Custom CNN + KNN", "MobileNetV2 + SVM",
    "EfficientNet-B0 + SVM", "EfficientNet-B0 + Softmax", "MobileNetV2 + Softmax",
    "Custom CNN + RF", "Custom CNN + Softmax", "Custom CNN + NB"
]
box_data = [sample_mre_dict[m] for m in box_models]
box_colors = [backbone_colors.get(m.split(" + ")[0], "#666666") for m in box_models]

bp = ax3.boxplot(box_data, vert=False, patch_artist=True, showmeans=True,
                 meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='red', markersize=5),
                 medianprops=dict(color='black', linewidth=1.5),
                 flierprops=dict(marker='o', markersize=2, alpha=0.25))

for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.set_yticks(range(1, len(box_models) + 1))
ax3.set_yticklabels(box_models, fontsize=8.5, fontweight="bold")
ax3.set_xlabel("Magnitude of Relative Error (MRE) per Sample", fontsize=10.5, fontweight="bold")
ax3.set_title("C. Sample-Level MRE Dispersion (Box: IQR, Line: Median, Red Diamond: Mean/MMRE)", fontsize=11.5, fontweight="bold", pad=10)
ax3.set_xlim(-0.02, 1.02)

# -----------------------------------------------------------------------------
# Panel D: Class-Level Prevalence MMRE vs Accuracy Pareto Frontier
# -----------------------------------------------------------------------------
for bb, color in backbone_colors.items():
    sub = df_results[df_results["Backbone"] == bb]
    ax4.scatter(sub["Class MMRE (%)"], sub["Test Acc (%)"], s=140, color=color, alpha=0.85, edgecolor="black", linewidth=1.2, label=bb)

for _, r in df_results.iterrows():
    # Label key models
    if r["Head"] in ["SVM", "Softmax", "KNN"] or r["Class MMRE (%)"] > 20:
        offset_x = 1.5 if r["Class MMRE (%)"] < 20 else -15.0
        offset_y = 0.4 if "Custom CNN + SVM" in r["Model"] else -0.8
        ax4.annotate(
            r["Model"],
            (r["Class MMRE (%)"], r["Test Acc (%)"]),
            xytext=(r["Class MMRE (%)"] + offset_x, r["Test Acc (%)"] + offset_y),
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85)
        )

ax4.set_xlabel("Class-Level Prevalence MMRE (%) [lower is better]", fontsize=10.5, fontweight="bold")
ax4.set_ylabel("Test Accuracy (%) [higher is better]", fontsize=10.5, fontweight="bold")
ax4.set_title("D. Accuracy vs. Class Prevalence Distortion (Quantification Fidelity)", fontsize=11.5, fontweight="bold", pad=10)
ax4.set_xlim(-2, 80)
ax4.set_ylim(65, 100)
ax4.legend(loc="lower left", frameon=True)

# Main Figure Title
fig.suptitle("Figure 16: Magnitude of Relative Error (MRE) & MMRE Architectural Benchmark Across 15 Models", fontsize=15, fontweight="bold", y=0.99)
plt.tight_layout()

fig_path = FIGURES_DIR / "fig16_mre_mmre_all_15_models.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)

downloads_path = Path("C:/Users/ritik/Downloads/fig16_mre_mmre_all_15_models.png")
if downloads_path.parent.exists():
    shutil.copy(fig_path, downloads_path)
    print(f"Copied figure to Downloads -> {downloads_path}")

print(f"\nFigure 16 successfully generated -> {fig_path}")

print("\n" + "=" * 80)
print("FINAL MRE & MMRE BENCHMARK SUMMARY (ALL 15 MODELS)")
print("=" * 80)
print(df_results.to_string(index=False))
print("=" * 80)

