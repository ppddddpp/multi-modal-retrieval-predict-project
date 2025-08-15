from dataParser import parse_openi_xml
from labeledData import disease_groups, normal_groups
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
from tensorDICOM import DICOMImagePreprocessor

# Resolve paths...
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

XML_DIR    = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
MODEL_PLACE = BASE_DIR / "models"
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_PLACE)

if __name__ == '__main__':
    # Load parsed data
    records = parse_openi_xml(XML_DIR, DICOM_ROOT)

    # Build label matrix from the new 'disease_vector' field
    label_matrix = np.array([rec['labels'] for rec in records])
    label_sums   = label_matrix.sum(axis=0)

    combined_groups = {
        **disease_groups,
        **normal_groups
    }

    # Get label names in the same order as the vector
    label_names = sorted(combined_groups.keys())

    # Plot normal vs abnormal
    normal_idx = label_names.index("Normal")
    n_strict_normal = sum(
        vec[normal_idx] == 1 and sum(vec) == 1
        for vec in label_matrix
    )
    n_abnormal = sum(
        any(vec[i] for i in range(len(vec)) if i != normal_idx)
        for vec in label_matrix
    )
    print(f"Strict Normal samples (only 'Normal' = 1): {n_strict_normal}")
    print(f"Abnormal samples (any disease group = 1): {n_abnormal}")

    plt.figure()
    plt.pie(
        [n_strict_normal, n_abnormal],
        labels=["Normal", "Abnormal"],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Normal vs Abnormal Cases")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # Plot total cases per disease
    plt.figure(figsize=(12,4))
    plt.bar(label_names, label_sums)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cases")
    plt.title("Disease Label Frequencies")
    plt.tight_layout()
    plt.show()

    # Count number of labels per image
    num_labels_per_sample = label_matrix.sum(axis=1)
    print("Average labels/image:", num_labels_per_sample.mean())
    print("Max labels on one image:", num_labels_per_sample.max())

    plt.figure()
    plt.hist(num_labels_per_sample, bins=np.arange(num_labels_per_sample.max()+2)-0.5, rwidth=0.8)
    plt.xlabel("Number of labels per image")
    plt.ylabel("Frequency")
    plt.title("Multi-label Distribution")
    plt.tight_layout()
    plt.show()

    # Check for shared reports
    report_map = defaultdict(list)
    for rec in records:
        report_map[rec['report_text']].append(rec['id'])

    shared = [ids for ids in report_map.values() if len(ids) > 1]
    print(f"Unique reports: {len(report_map)}")
    print(f"Reports shared by multiple images: {len(shared)}")
    print(f"Avg images per reused report: {np.mean([len(ids) for ids in shared]):.2f}")

    # Debug a sample DICOM
    dp = DICOMImagePreprocessor()
    sample = records[0]
    arr = dp.load_raw_array(sample['dicom_path'])

    plt.figure()
    plt.imshow(arr, cmap="gray")
    plt.title("DICOM Image")
    plt.axis("off")
    plt.show()

    # Print sample report and its labels
    print("--- Report ---")
    print(sample["report_text"])
    print("\n--- Labels ---")
    print({
        name: val
        for name, val in zip(label_names, sample["labels"])
        if val
    })

    df = pd.DataFrame({
        'id':        [rec['id'] for rec in records],
        'text':      [rec['report_text'] for rec in records],
        **{name: [vec[i] for vec in (rec['labels'] for rec in records)] 
        for i, name in enumerate(label_names)}
    })

    # Label prevalence bar chart
    label_cols = label_names
    label_counts = df[label_cols].sum().sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x=label_counts.values, y=label_counts.index)
    plt.title("Label Prevalence (N reports)")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.show()

    # Histogram of labels per report
    df['num_labels'] = df[label_cols].sum(axis=1)
    plt.figure(figsize=(6,4))
    sns.histplot(df['num_labels'], bins=range(0, int(df['num_labels'].max())+2), discrete=True)
    plt.title("Labels per Report")
    plt.xlabel("Number of Labels")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Label co-occurrence heatmap
    co_mat = df[label_cols].T.dot(df[label_cols])
    plt.figure(figsize=(12,10))
    sns.heatmap(co_mat, annot=False, cmap="Blues", fmt="d")
    plt.title("Label Co-occurrence Matrix")
    plt.tight_layout()
    plt.show()

    # Report length analysis
    #    compute word count
    df['word_count'] = df['text'].str.split().map(len)

    # overall distribution
    plt.figure(figsize=(6,4))
    sns.histplot(df['word_count'], bins=30)
    plt.title("Report Word Count Distribution")
    plt.xlabel("Words per Report")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # boxplot of length by key labels (choose top 5 frequent)
    top5 = label_counts.index[:5].tolist()

    # Melt the DataFrame so each row is (word_count, label, has_label)
    melted = pd.melt(
        df,
        id_vars=['word_count'],
        value_vars=top5,
        var_name='label',
        value_name='has_label'
    )

    # Keep only the rows where the label is present
    melted_pos = melted[melted['has_label'] == 1]

    # Plot boxplots of word_count for each of the top-5 labels
    plt.figure(figsize=(10,6))
    sns.boxplot(
        data=melted_pos,
        x='label',
        y='word_count'
    )
    plt.title("Report Length by Top-5 Labels (only positive cases)")
    plt.xlabel("Label")
    plt.ylabel("Words per Report")
    plt.tight_layout()
    plt.show()

    # Positive:Negative ratio per class
    N = len(df)
    pos_neg = pd.DataFrame({
        'label': label_counts.index,
        'pos':   label_counts.values,
        'neg':   N - label_counts.values
    })
    pos_neg['pos_neg_ratio'] = pos_neg['pos'] / pos_neg['neg']

    plt.figure(figsize=(10,6))
    sns.barplot(x='pos_neg_ratio', y='label', data=pos_neg.sort_values('pos_neg_ratio'))
    plt.title("Positive:Negative Ratio by Label")
    plt.xlabel("Positive / Negative")
    plt.tight_layout()
    plt.show()

    # Print summary table
    print("\nClass imbalance summary:")
    print(pos_neg.sort_values('pos_neg_ratio').to_string(index=False, float_format="%.3f"))