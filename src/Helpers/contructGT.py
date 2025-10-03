from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import json
import pandas as pd
from datetime import datetime
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups

BASE_DIR     = Path(__file__).resolve().parent.parent.parent
SPLIT_DIR    = BASE_DIR / "splited_data"
GT_DIR       = BASE_DIR / "ground_truths"
GT_DIR.mkdir(exist_ok=True)

def create_gt(split_dir=None, gt_save_dir=None, 
                log_file="gt_construct_log.txt" ,combined_groups=None):
    """
    Creates a ground truth file for generalization and historical relevance.
    Ground truth files are saved in the GT_DIR directory by default.

    Parameters
    ----------
    split_dir : str
        Path to the directory containing the split IDs JSON files
    gt_save_dir : str
        Path to the directory where the ground truth files will be saved
    log_file : str
        Path to the log file
    combined_groups : dict
        Dictionary where keys are disease/normal group names and values are lists of labels

    Returns
    -------
    None

    Saves two JSON files: test_relevance.json and test_to_train_relevance.json
    """
    if combined_groups is not None:
        print("Using provided combined_groups")
    else:
        combined_groups = {**disease_groups, **normal_groups, **finding_groups, **symptom_groups}

    if split_dir is None:
        split_dir = SPLIT_DIR
    
    with open(split_dir / "test_split_ids.json") as f:
        test_ids = json.load(f)
    with open(split_dir / "train_split_ids.json") as f:
        train_ids = json.load(f)

    df_test  = pd.read_csv(split_dir / "openi_test_labeled.csv")
    df_test  = df_test[df_test["id"].isin(test_ids)].reset_index(drop=True)

    df_train = pd.read_csv(split_dir / "openi_train_labeled.csv")
    df_train = df_train[df_train["id"].isin(train_ids)].reset_index(drop=True)

    # Ensure the label columns are consistent with disease and normal groups
    label_cols = list(combined_groups.keys())
    test_vals  = df_test[label_cols].values.astype(int)   # shape (N_test, L)
    test_ids    = df_test["id"].tolist()

    train_vals = df_train[label_cols].values.astype(int)  # shape (N_train, L)
    train_ids   = df_train["id"].tolist()

    test_relevance = {}
    for i, qid in enumerate(test_ids):
        # boolean mask of same-label examples
        shared = (test_vals & test_vals[i]).sum(axis=1) > 0
        # exclude the query itself
        rel_ids = [test_ids[j] for j, keep in enumerate(shared) if keep and j != i]
        test_relevance[qid] = rel_ids

    # Build test→train relevance (historical)
    test_to_train = {}
    for i, qid in enumerate(test_ids):
        shared = (train_vals & test_vals[i]).sum(axis=1) > 0
        rel_ids = [train_ids[j] for j, keep in enumerate(shared) if keep]
        test_to_train[qid] = rel_ids

    rel_counts = [len(v) for v in test_relevance.values()]
    min_rel = min(rel_counts)
    max_rel = max(rel_counts)
    avg_rel = sum(rel_counts) / len(rel_counts)

    print("Min relevant items per query:", min_rel)
    print("Max relevant items per query:", max_rel)
    print("Avg relevant items per query:", avg_rel)

    if gt_save_dir is None:
        gt_save_dir = GT_DIR

    with open(gt_save_dir / "test_relevance.json", "w") as f:
        json.dump(test_relevance, f, indent=2)

    with open(gt_save_dir / "test_to_train_relevance.json", "w") as f:
        json.dump(test_to_train, f, indent=2)

    msg_general = f"Saved generalization GT -> {gt_save_dir/'test_relevance.json'}"
    msg_hist    = f"Saved historical  GT -> {gt_save_dir/'test_to_train_relevance.json'}"

    print(msg_general)
    print(msg_hist)

    # --- Save log ---
    log_path = gt_save_dir / log_file
    with open(log_path, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(f"Min relevant items per query: {min_rel}\n")
        f.write(f"Max relevant items per query: {max_rel}\n")
        f.write(f"Avg relevant items per query: {avg_rel:.2f}\n")
        f.write(msg_general + "\n")
        f.write(msg_hist + "\n\n")

    print(f"Logged summary to {log_path}")

if __name__ == "__main__":
    create_gt()