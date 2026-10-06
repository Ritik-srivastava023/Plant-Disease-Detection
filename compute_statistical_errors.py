"""Generate a research-ready statistical comparison from the existing model artifacts.

The project currently has one confirmed test split and a CSV containing reported
variation statistics. This script therefore reports the existing mean ± standard
variation as *reported variability*, not as a statistically independent
replication result. A future extension can replace these values with true
multi-seed results.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
RESULTS_FILE = ROOT / "all_15_models_metrics.csv"
STATISTICS_FILE = ARTIFACTS_DIR / "statistical_error_metrics.csv"
OUTPUT_TABLE = ARTIFACTS_DIR / "research_statistical_summary.csv"
OUTPUT_FIGURE = FIGURES_DIR / "fig13_research_statistical_summary.png"


def normalize_head(value: str) -> str:
    """Normalize classifier names used by different scripts."""
    value = str(value).strip()
    replacements = {
        "SVM (RBF)": "SVM",
        "KNN (k=5)": "KNN",
        "Random Forest": "RF",
        "Gaussian NB": "NB",
    }
    return replacements.get(value, value)


def load_and_merge() -> pd.DataFrame:
    """Merge the verified metrics with the reported variability values."""
    results = pd.read_csv(RESULTS_FILE)
    statistics = pd.read_csv(STATISTICS_FILE)
    statistics["Head"] = statistics["Head"].map(normalize_head)

    columns = [
        "Model",
        "Backbone",
        "Head",
        "Test Acc (%)",
        "Acc SD (%)",
        "Mean Error Rate (%)",
        "Error SD (%)",
        "Macro F1 (%)",
        "F1 SD (%)",
        "Mean Absolute Error (MAE)",
        "MAE SD",
        "Latency (ms)",
    ]

    benchmark = results[["Model", "Backbone", "Classifier", "Test Acc (%)", "Macro F1 (%)", "Latency (ms)"]].rename(
        columns={"Classifier": "Head"}
    )
    statistics = statistics.rename(columns={"Head": "Head"})
    merged = benchmark.merge(
        statistics,
        on=["Backbone", "Head"],
        how="inner",
        validate="one_to_one",
    )

    select = {
        "Model": merged["Model_x"],
        "Backbone": merged["Backbone"],
        "Head": merged["Head"],
        "Test Acc (%)": merged["Test Acc (%)_x"],
        "Acc SD (%)": merged["Acc SD (%)"],
        "Mean Error Rate (%)": merged["Mean Error Rate (%)"],
        "Error SD (%)": merged["Error SD (%)"],
        "Macro F1 (%)": merged["Macro F1 (%)_x"],
        "F1 SD (%)": merged["F1 SD (%)"],
        "Mean Absolute Error (MAE)": merged["Mean Absolute Error (MAE)"],
        "MAE SD": merged["MAE SD"],
        "Latency (ms)": merged["Latency (ms)"],
    }
    return pd.DataFrame(select).sort_values("Test Acc (%)", ascending=False).reset_index(drop=True)


def make_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Build a compact table suitable for a paper and presentation slide."""
    return summary[
        [
            "Model",
            "Backbone",
            "Head",
            "Test Acc (%)",
            "Acc SD (%)",
            "Mean Error Rate (%)",
            "Error SD (%)",
            "Macro F1 (%)",
            "F1 SD (%)",
            "Latency (ms)",
        ]
    ].copy()


def plot_summary(summary: pd.DataFrame) -> None:
    """Create a four-panel comparison graph."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    palette = {
        "Softmax": "#d95f02",
        "SVM": "#1f77b4",
        "KNN": "#2ca02c",
        "RF": "#9467bd",
        "NB": "#8c564b",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    summary["x"] = range(len(summary))

    for ax, metric, title, y_label, lower, upper in (
        (axes[0, 0], "Test Acc (%)", "A. Accuracy and reported variability", "Accuracy (%)", 50, 100),
        (axes[0, 1], "Macro F1 (%)", "B. Macro F1 and reported variability", "Macro F1 (%)", 50, 100),
        (axes[1, 0], "Mean Error Rate (%)", "C. Mean error rate", "Error rate (%)", 0, 35),
        (axes[1, 1], "Latency (ms)", "D. Inference latency", "Latency (ms/image)", 0, None),
    ):
        colors = [palette.get(row["Head"], "#666666") for _, row in summary.iterrows()]
        y_error = None
        if metric == "Test Acc (%)":
            y_error = summary["Acc SD (%)"]
        elif metric == "Macro F1 (%)":
            y_error = summary["F1 SD (%)"]

        ax.bar(
            summary["x"],
            summary[metric],
            color=colors,
            edgecolor="black",
            linewidth=0.7,
            yerr=y_error,
            capsize=4,
            alpha=0.9,
        )

        for x, value in zip(summary["x"], summary[metric]):
            ax.text(
                x,
                value + (1 if metric != "Latency (ms)" else max(value * 0.05, 1.0)),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                rotation=90,
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=10, fontweight="bold")
        ax.set_ylim(lower, upper)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axis_labels = [
        f"{row['Model'].split(' + ')[0]}\n{row['Head']}"
        for _, row in summary.iterrows()
    ]
    axes[1, 0].set_xticks(summary["x"])
    axes[1, 0].set_xticklabels(axis_labels, rotation=60, ha="right", fontsize=7)
    axes[1, 1].set_xticks(summary["x"])
    axes[1, 1].set_xticklabels(axis_labels, rotation=60, ha="right", fontsize=7)

    for ax in axes.flat:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Plant Disease Classification: 15-Model Comparison with Reported Variability",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Reported variability is the SD from the existing artifact; it is not evidence of independent repeated runs.",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.96))
    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_paper_summary(summary: pd.DataFrame) -> None:
    """Print the top two models and the most important improvement facts."""
    best_accuracy = summary.iloc[0]
    best_f1 = summary.sort_values("Macro F1 (%)", ascending=False).iloc[0]
    most_compact = summary.sort_values("Latency (ms)", ascending=True).iloc[0]

    print("\n=== RESEARCH SUMMARY ===")
    print(f"Highest accuracy: {best_accuracy['Model']} = {best_accuracy['Test Acc (%)']:.2f}%")
    print(f"Highest macro F1: {best_f1['Model']} = {best_f1['Macro F1 (%)']:.2f}%")
    print(f"Lowest latency: {most_compact['Model']} = {most_compact['Latency (ms)']:.3f} ms/image")
    print(f"Figure saved to: {OUTPUT_FIGURE}")


def main() -> None:
    summary = load_and_merge()
    table = make_summary_table(summary)
    table.to_csv(OUTPUT_TABLE, index=False)
    plot_summary(summary)
    print_paper_summary(summary)


if __name__ == "__main__":
    main()
