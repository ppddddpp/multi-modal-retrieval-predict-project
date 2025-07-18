import pandas as pd
from pathlib import Path
from labeledData import disease_groups, normal_groups
import matplotlib.pyplot as plt
import seaborn as sns

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent
    
if __name__ == '__main__':
    # Load original & final
    orig  = pd.read_csv(BASE_DIR / "outputs" / "openi_labels.csv")
    final = pd.read_csv(BASE_DIR / "outputs" / "openi_labels_final.csv")

    # Define one‑hot columns
    label_cols = list({**disease_groups, **normal_groups}.keys())

    final["llm_status"] = final["llm_status"].astype(str).str.strip().str.title()
    status_counts = final["llm_status"].value_counts()
    delta = final[label_cols] - orig[label_cols] # Compute delta = final - orig

    # Summarize absolute counts
    summary = pd.DataFrame({
        "added":     (delta == 1).sum(),
        "removed":   (delta == -1).sum(),
        "unchanged": (delta == 0).sum(),
    })

    # Compute percents relative to total reports
    total = len(delta)
    summary["% added"]   = summary["added"]   / total * 100
    summary["% removed"] = summary["removed"] / total * 100

    # Net-change just for reference
    summary["net_change"] = summary["added"] - summary["removed"]
    summary = summary.sort_values("% added", ascending=False)

    # count how many 1’s per row
    final["n_labels"] = final[label_cols].sum(axis=1)

    # summary stats
    print(final["n_labels"].describe())
    print(summary[["added","% added","removed","% removed","net_change"]])
    print(status_counts)

    # --- Plotting ---

    # Percentage added vs removed (horizontal bar chart)
    fig, ax = plt.subplots(figsize=(8, max(4, len(label_cols)*0.2)))
    summary[["% added", "% removed"]].plot.barh(ax=ax)
    ax.set_xlabel("% of reports")
    ax.set_title("Percentage of Reports Where Each Label Was Added or Removed")
    plt.tight_layout()
    plt.show()

    # Net count change (horizontal bar chart)
    fig, ax2 = plt.subplots(figsize=(8, max(4, len(label_cols)*0.2)))
    summary["net_change"].sort_values().plot.barh(ax=ax2)
    ax2.set_xlabel("Net change (added – removed)")
    ax2.set_title("Net Count Change by Label")
    plt.tight_layout()
    plt.show()

    # LLM Status Distribution
    plt.figure(figsize=(6,5))
    status_counts.plot.bar()
    plt.title("LLM Status Distribution")
    plt.ylabel("Count")
    plt.xlabel("llm_status")
    plt.tight_layout()
    plt.show()

    # histogram
    cooc = final[label_cols].T.dot(final[label_cols])
    plt.figure(figsize=(8, 5))
    final["n_labels"].hist(bins=range(final["n_labels"].max() + 2))
    plt.title("Distribution of Number of Labels per Report")
    plt.xlabel("Number of Labels")
    plt.ylabel("Number of Reports")
    plt.tight_layout()
    plt.show()

    # percentage of reports with label i that also have j
    cooc_norm = cooc.div(cooc.values.diagonal(), axis=0)
    plt.figure(figsize=(12,10))
    sns.heatmap(cooc_norm, cmap="viridis", xticklabels=False, yticklabels=False)
    plt.title("Label Co-occurrence (normalized)")
    plt.show()