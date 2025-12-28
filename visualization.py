import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_poster_visuals():
    """
    Generates the confusion matrix and accuracy comparison charts
    based on the research poster results.
    """
    print("\nGenerating Research Poster Visualizations...")

    # ---------------- Confusion Matrix ----------------
    cm_data = np.array([[450, 85], [36, 120]])
    labels = ["Literal\n(Blue)", "Sarcastic\n(Orange)"]

    annot_labels = [
        ["450\n(True Negative)", "85\n(False Positive)"],
        ["36\n(False Negative\nMissed Sarcasm)", "120\n(True Positive)"]
    ]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_data,
        annot=annot_labels,
        fmt="",
        cmap="Oranges",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 12, "weight": "bold"}
    )

    plt.xlabel("Predicted ML Label", fontweight="bold")
    plt.ylabel("Actual Linguistic Meaning", fontweight="bold")
    plt.title("Quantifying Linguistic Failure (Confusion Matrix)",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # ---------------- Accuracy Comparison ----------------
    categories = ["Linguistic Limitations", "High Accuracy"]
    values = [78, 98]
    colors = ["#FFC000", "#0070C0"]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(categories, values, color=colors, width=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height - 10,
            f"{height}%",
            ha="center",
            va="bottom",
            color="white",
            fontweight="bold",
            fontsize=14
        )

    plt.title("Performance Gap: Context Matters",
              fontsize=14, fontweight="bold")
    plt.ylabel("Percentage")
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.show()

    print("Poster visualizations displayed successfully.")
