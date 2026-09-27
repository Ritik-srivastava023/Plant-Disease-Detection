"""
Benchmark Evaluation & Latency Comparison for Plant Disease Classification.
Generates research comparison tables (CSV, Markdown, LaTeX) and publication figures.
"""
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB0

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Architecture Builders to Measure Parameters and Latency
# -------------------------------------------------------------
def build_custom_cnn():
    return models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(15, activation='softmax')
    ])

def build_mobilenet():
    base = MobileNetV2(weights=None, include_top=False, input_shape=(128, 128, 3))
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(15, activation='softmax')(x)
    return models.Model(inputs=base.input, outputs=out)

def build_efficientnet():
    base = EfficientNetB0(weights=None, include_top=False, input_shape=(128, 128, 3))
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(15, activation='softmax')(x)
    return models.Model(inputs=base.input, outputs=out)

# -------------------------------------------------------------
# 2. Benchmark Inference Latency & Model Sizes
# -------------------------------------------------------------
def measure_latency_and_params(model, input_shape=(1, 128, 128, 3), runs=60, warmup=10):
    dummy = np.random.randn(*input_shape).astype(np.float32)
    # Warmup
    for _ in range(warmup):
        _ = model(dummy, training=False)
    
    # Timing
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(dummy, training=False)
    t1 = time.perf_counter()
    latency_ms = ((t1 - t0) / runs) * 1000.0
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    return total_params, trainable_params, latency_ms

print("=== Benchmarking Architectures & Latency ===")
models_dict = {
    "Custom CNN (GAP + BN)": (build_custom_cnn(), 0.48),
    "MobileNetV2 (Two-Stage)": (build_mobilenet(), 11.51),
    "EfficientNet-B0 (Two-Stage)": (build_efficientnet(), 17.50),
}

benchmark_data = []

# Load real test predictions if available
test_preds = {}
for mname, fname in [
    ("Custom CNN (GAP + BN)", "cnn_test_predictions.csv"),
    ("MobileNetV2 (Two-Stage)", "mobilenet_test_predictions.csv"),
    ("EfficientNet-B0 (Two-Stage)", "efficientnet_test_predictions.csv")
]:
    p = ROOT / fname
    if p.exists():
        df = pd.read_csv(p)
        test_preds[mname] = df

for name, (m, disk_size_mb) in models_dict.items():
    tot_params, train_params, latency_ms = measure_latency_and_params(m)
    
    if name in test_preds:
        df = test_preds[name]
        acc = accuracy_score(df['true_label'], df['predicted_label']) * 100
        macro_prec = precision_score(df['true_label'], df['predicted_label'], average='macro', zero_division=0) * 100
        macro_rec = recall_score(df['true_label'], df['predicted_label'], average='macro', zero_division=0) * 100
        macro_f1 = f1_score(df['true_label'], df['predicted_label'], average='macro', zero_division=0) * 100
        weighted_f1 = f1_score(df['true_label'], df['predicted_label'], average='weighted', zero_division=0) * 100

        # Calculate authentic One-vs-Rest ROC-AUC dynamically from full probability vectors
        prob_cols = [col for col in df.columns if col.startswith("prob_")]
        if prob_cols:
            classes_from_cols = [c.replace("prob_", "") for c in prob_cols]
            y_true_indices = np.array([classes_from_cols.index(label) for label in df['true_label']])
            y_true_onehot = label_binarize(y_true_indices, classes=np.arange(len(classes_from_cols)))
            probabilities = df[prob_cols].to_numpy()
            macro_auc = roc_auc_score(y_true_onehot, probabilities, average='macro', multi_class='ovr')
        else:
            raise ValueError(f"No probability columns found in prediction CSV for {name}.")
    else:
        raise FileNotFoundError(f"Missing prediction CSV for model: {name}")

    benchmark_data.append({
        "Model": name,
        "Paradigm": "Scratch Baseline" if "Custom" in name else ("Edge Lightweight" if "Mobile" in name else "High-Capacity SOTA"),
        "Total Params": tot_params,
        "Params (M)": round(tot_params / 1e6, 3),
        "Size (MB)": disk_size_mb,
        "CPU Latency (ms/img)": round(latency_ms, 2),
        "Throughput (img/s)": round(1000.0 / latency_ms, 1),
        "Test Accuracy (%)": round(acc, 2),
        "Macro Precision (%)": round(macro_prec, 2),
        "Macro Recall (%)": round(macro_rec, 2),
        "Macro F1 (%)": round(macro_f1, 2),
        "Weighted F1 (%)": round(weighted_f1, 2),
        "Macro OvR AUC": round(macro_auc, 4),
    })

summary_df = pd.DataFrame(benchmark_data)
csv_out = OUTPUT_DIR / "model_comparison_table.csv"
summary_df.to_csv(csv_out, index=False)
print(f"Saved benchmark summary to {csv_out}")

# -------------------------------------------------------------
# 3. Generate Markdown & LaTeX Tables for Final Presentation
# -------------------------------------------------------------
markdown_table = summary_df[[
    "Model", "Paradigm", "Params (M)", "Size (MB)", "CPU Latency (ms/img)",
    "Test Accuracy (%)", "Macro F1 (%)", "Weighted F1 (%)", "Macro OvR AUC"
]].to_markdown(index=False)

md_out = OUTPUT_DIR / "benchmark_table.md"
md_out.write_text(markdown_table, encoding="utf-8")

latex_table = summary_df[[
    "Model", "Params (M)", "Size (MB)", "CPU Latency (ms/img)",
    "Test Accuracy (%)", "Macro F1 (%)", "Weighted F1 (%)", "Macro OvR AUC"
]].to_latex(index=False, float_format="%.2f")

tex_out = OUTPUT_DIR / "benchmark_table.tex"
tex_out.write_text(latex_table, encoding="utf-8")

print("\n" + "="*80)
print("BENCHMARK COMPARISON TABLE (FOR REPORT & PRESENTATION)")
print("="*80)
print(markdown_table)
print("="*80)

# -------------------------------------------------------------
# 4. Generate Publication-Quality Comparison Figure
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: Macro F1 vs Latency (Trade-Off Frontier)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

for i, row in summary_df.iterrows():
    axes[0].scatter(
        row['CPU Latency (ms/img)'], row['Macro F1 (%)'],
        s=row['Size (MB)'] * 30 + 100, color=colors[i], marker=markers[i],
        label=f"{row['Model']} ({row['Size (MB)']} MB)", alpha=0.85, edgecolors='black', linewidth=1.5
    )
    axes[0].annotate(
        row['Model'].split()[0],
        (row['CPU Latency (ms/img)'] + 1.5, row['Macro F1 (%)'] - 0.3),
        fontsize=10, fontweight='bold'
    )

axes[0].set_title("Edge Efficiency: Macro F1 vs. Latency (Bubble = Size)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("CPU Inference Latency (ms / image)", fontsize=11)
axes[0].set_ylabel("Macro F1 Score (%)", fontsize=11)
axes[0].set_ylim(55, 102)
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(loc="lower right", fontsize=9)

# Subplot 2: Model Size and Parameter Reduction
bar_width = 0.35
indices = np.arange(len(summary_df))
axes[1].bar(indices - bar_width/2, summary_df['Params (M)'], width=bar_width, label="Parameters (Millions)", color='#4c72b0')
axes[1].bar(indices + bar_width/2, summary_df['Size (MB)'], width=bar_width, label="Disk Size (MB)", color='#55a868')
axes[1].set_xticks(indices)
axes[1].set_xticklabels([m.split()[0] for m in summary_df['Model']], fontsize=10, fontweight='bold')
axes[1].set_title("Model Complexity & Footprint", fontsize=12, fontweight='bold')
axes[1].set_ylabel("Count / MB", fontsize=11)
axes[1].legend(loc="upper left")
axes[1].grid(True, linestyle="--", alpha=0.4, axis='y')

for i in indices:
    axes[1].text(i - bar_width/2, summary_df['Params (M)'].iloc[i] + 0.1, f"{summary_df['Params (M)'].iloc[i]:.2f}M", ha='center', fontsize=9)
    axes[1].text(i + bar_width/2, summary_df['Size (MB)'].iloc[i] + 0.3, f"{summary_df['Size (MB)'].iloc[i]:.1f}MB", ha='center', fontsize=9)

# Subplot 3: Diagnostic Metrics Breakdown
metrics = ['Test Accuracy (%)', 'Macro F1 (%)', 'Weighted F1 (%)']
x = np.arange(len(metrics))
width = 0.25

for i, row in summary_df.iterrows():
    vals = [row[m] for m in metrics]
    axes[2].bar(x + (i - 1)*width, vals, width=width, label=row['Model'].split()[0], color=colors[i])

axes[2].set_xticks(x)
axes[2].set_xticklabels(['Accuracy', 'Macro F1', 'Weighted F1'], fontsize=10, fontweight='bold')
axes[2].set_ylim(55, 102)
axes[2].set_ylabel("Score (%)", fontsize=11)
axes[2].set_title("Performance Metrics on Untouched Test Split", fontsize=12, fontweight='bold')
axes[2].legend(loc="lower right")
axes[2].grid(True, linestyle="--", alpha=0.4, axis='y')

plt.tight_layout()
fig_out = OUTPUT_DIR / "model_comparison.png"
plt.savefig(fig_out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved publication comparison figure to {fig_out}")

