from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
from DataHandler import DICOMImagePreprocessor, parse_openi_xml
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups

# Resolve paths...
try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent.parent

XML_DIR     = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT  = BASE_DIR / 'data' / 'openi' / 'dicom'
OUTPUT_FILE = BASE_DIR / "outputs" / "openi_labels_final.csv"
OUTPUT_DROP_ZERO = BASE_DIR / "outputs"
MODEL_PLACE = BASE_DIR / "models"
EDA_DIR     = BASE_DIR / "eda_data"
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_PLACE)


def get_eda_before_split(xml_dir=XML_DIR, dicom_root=DICOM_ROOT, eda_dir=EDA_DIR,
                            output_file=OUTPUT_FILE, combine_groups=None,
                            drop_zero=True, save_cleaned=True,
                            max_show=10, output_drop_zero=OUTPUT_DROP_ZERO     
                            ):
    """
    Compute various statistics and plots for the OpenI dataset.

    Parameters
    ----------
    xml_dir : str, optional
        Path to the OpenI XML directory.
    dicom_root : str, optional
        Path to the OpenI DICOM directory.
    eda_dir : str, optional
        Path to the directory where the EDA plots will be saved.
    output_file : str, optional
        Path to the file which has the corrected labels generated and mapped correctly after llm verification.
    combine_groups : dict, optional
        Dictionary mapping label group names to their corresponding sub-labels.
    drop_zero : bool, optional
        If True, drop the zero-label reports and save the cleaned dataset.
    save_cleaned : bool, optional
        If True, save the cleaned dataset (with zero-label reports dropped).
    max_show : int, optional
        Maximum number of reports to show for each statistic.
    output_drop_zero : str, optional
        Path to the directory where the cleaned dataset backup will be saved.

    Returns
    -------
    None

    Notes
    -----
    This function assumes that the OpenI dataset has been preprocessed and that the corrected labels have been saved to a CSV file.

    The function first loads the corrected labels and computes the label prevalence, then plots a bar chart of the label prevalence. It then computes the label co-occurrence matrix and plots a heatmap of the matrix. Finally, it computes the report length distribution and plots a histogram of the distribution.

    If drop_zero is True, the function drops the zero-label reports and saves the cleaned dataset to a CSV file. If save_cleaned is True, the function also saves a backup of the dataset with zero-label reports to a CSV file.

    The function returns None.
    """

    eda_dir.mkdir(parents=True, exist_ok=True)
    log_file = eda_dir / "eda_before_split_summary.txt"

    # utility to log and print
    def log_and_print(msg=""):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    # clear old log file
    open(log_file, "w").close()

    combined_groups_temp = {
        **disease_groups,
        **finding_groups,
        **symptom_groups,
        **normal_groups
    }
    combined_groups = combined_groups_temp if combine_groups is None else combine_groups
    label_names = sorted(combined_groups.keys())

    # --------------------------
    # Load data
    # --------------------------
    if output_file.exists():
        log_and_print(f"[INFO] Using corrected labels from {output_file}")
        df = pd.read_csv(output_file)
        label_matrix = df[label_names].values
        use_records = False
    else:
        log_and_print("[INFO] Corrected CSV not found, falling back to parsing XML")
        records = parse_openi_xml(xml_dir, dicom_root, combined_groups=combined_groups)
        label_matrix = np.array([rec['labels'] for rec in records])
        df = pd.DataFrame({
            'id':          [rec['id'] for rec in records],
            'report_text': [rec['report_text'] for rec in records],
            **{name: [vec[i] for vec in (rec['labels'] for rec in records)]
               for i, name in enumerate(label_names)}
        })
        use_records = True
    
    if "report_text" in df.columns:
        df.rename(columns={"report_text": "text"}, inplace=True)

    # --------------------------
    # Normal vs abnormal counts
    # --------------------------
    normal_idx = label_names.index("Normal")
    n_strict_normal = sum(
        vec[normal_idx] == 1 and sum(vec) == 1
        for vec in label_matrix
    )
    n_abnormal = sum(
        any(vec[i] for i in range(len(vec)) if i != normal_idx)
        for vec in label_matrix
    )
    log_and_print(f"Strict Normal samples (only 'Normal' = 1): {n_strict_normal}")
    log_and_print(f"Abnormal samples (any disease group = 1): {n_abnormal}")

    plt.figure()
    plt.pie([n_strict_normal, n_abnormal],
            labels=["Normal", "Abnormal"],
            autopct="%1.1f%%", startangle=90)
    plt.title("Normal vs Abnormal Cases")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(eda_dir / "normal_vs_abnormal.png")
    plt.close()

    if use_records:
        # Check for shared reports
        report_map = defaultdict(list)
        for rec in records:
            report_map[rec['report_text']].append(rec['id'])

        shared = [ids for ids in report_map.values() if len(ids) > 1]
        log_and_print(f"Unique reports: {len(report_map)}")
        log_and_print(f"Reports shared by multiple images: {len(shared)}")
        log_and_print(f"Avg images per reused report: {np.mean([len(ids) for ids in shared]):.2f}")

        # Debug a sample DICOM
        dp = DICOMImagePreprocessor()
        sample = records[0]
        arr = dp.load_raw_array(sample['dicom_path'])

        plt.figure()
        plt.imshow(arr, cmap="gray")
        plt.title("DICOM Image")
        plt.axis("off")
        plt.show()

        # Print sample report + labels
        log_and_print("--- Report ---")
        log_and_print(sample["report_text"])
        log_and_print("\n--- Labels ---")
        log_and_print({name: val for name, val in zip(label_names, sample["labels"]) if val})

    # Label prevalence bar chart
    label_cols = label_names
    label_counts = df[label_cols].sum().sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x=label_counts.values, y=label_counts.index)
    plt.title("Label Prevalence (N reports)")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(eda_dir / "label_prevalence.png")
    plt.close()

    # Histogram of labels per report
    df['num_labels'] = df[label_cols].sum(axis=1)
    plt.figure(figsize=(6,4))
    sns.histplot(df['num_labels'],
                 bins=range(0, int(df['num_labels'].max())+2), discrete=True)
    plt.title("Labels per Report")
    plt.xlabel("Number of Labels")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(eda_dir / "labels_per_report.png")
    plt.close()

    # Label co-occurrence heatmap
    co_mat = df[label_cols].T.dot(df[label_cols])
    plt.figure(figsize=(12,10))
    sns.heatmap(co_mat, annot=False, cmap="Blues", fmt="d")
    plt.title("Label Co-occurrence Matrix")
    plt.tight_layout()
    plt.savefig(eda_dir / "label_cooccurrence.png")
    plt.close()

    # Report length analysis
    df['word_count'] = df['text'].str.split().map(len)
    plt.figure(figsize=(6,4))
    sns.histplot(df['word_count'], bins=30)
    plt.title("Report Word Count Distribution")
    plt.xlabel("Words per Report")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(eda_dir / "report_word_count.png")
    plt.close()

    # Boxplot for top-5 frequent labels
    top5 = label_counts.index[:5].tolist()
    melted = pd.melt(df, id_vars=['word_count'], value_vars=top5,
                     var_name='label', value_name='has_label')
    melted_pos = melted[melted['has_label'] == 1]

    plt.figure(figsize=(10,6))
    sns.boxplot(data=melted_pos, x='label', y='word_count')
    plt.title("Report Length by Top-5 Labels (only positive cases)")
    plt.xlabel("Label")
    plt.ylabel("Words per Report")
    plt.tight_layout()
    plt.savefig(eda_dir / "report_length_by_label.png")
    plt.close()

    # Positive:Negative ratio
    N = len(df)
    pos_neg = pd.DataFrame({
        'label': label_counts.index,
        'pos':   label_counts.values,
        'neg':   N - label_counts.values
    })
    pos_neg['pos_neg_ratio'] = pos_neg['pos'] / pos_neg['neg']

    plt.figure(figsize=(10,6))
    sns.barplot(x='pos_neg_ratio', y='label',
                data=pos_neg.sort_values('pos_neg_ratio'))
    plt.title("Positive:Negative Ratio by Label")
    plt.xlabel("Positive / Negative")
    plt.tight_layout()
    plt.savefig(eda_dir / "pos_neg_ratio.png")
    plt.close()

    log_and_print("\nClass imbalance summary:")
    log_and_print(pos_neg.sort_values('pos_neg_ratio').to_string(index=False, float_format="%.3f"))

    # Print out the zero‐label reports 
    zero_idxs = df.index[df['num_labels'] == 0].tolist()
    log_and_print(f"\nFound {len(zero_idxs)} reports with 0 labels. Showing up to 10 of them:\n")

    max_show = 10
    for idx in zero_idxs[:max_show]:
        row = df.loc[idx]
        log_and_print(f"--- Report ID: {row['id']} (index={idx}) ---")
        log_and_print(row['text'])
        log_and_print("-" * 80)

    # Optional: drop and save
    if drop_zero and len(zero_idxs) > 0:
        log_and_print(f"\n[INFO] Dropping {len(zero_idxs)} zero-label reports.")

        # Keep a backup before dropping
        if save_cleaned:
            backup_path = output_drop_zero / "openi_labels_final_with_zero.csv"
            df.to_csv(backup_path, index=False)
            log_and_print(f"[INFO] Backup with zero-labels saved to {backup_path}")

        # Drop and reset index
        df = df[df['num_labels'] > 0].reset_index(drop=True)

        if save_cleaned:
            cleaned_path = output_drop_zero / "openi_labels_final_cleaned.csv"
            df.to_csv(cleaned_path, index=False)
            log_and_print(f"[INFO] Cleaned dataset saved to {cleaned_path}")
