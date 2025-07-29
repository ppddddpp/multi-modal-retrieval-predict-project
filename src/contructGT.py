import json
import pandas as pd
from pathlib import Path
from labeledData import disease_groups, normal_groups

BASE_DIR     = Path(__file__).resolve().parent.parent
SPLIT_DIR    = BASE_DIR / "splited_data"
GT_DIR       = BASE_DIR / "ground_truths"
GT_DIR.mkdir(exist_ok=True)

with open(SPLIT_DIR / "test_split_ids.json") as f:
    test_ids = json.load(f)
with open(SPLIT_DIR / "train_split_ids.json") as f:
    train_ids = json.load(f)

df_test  = pd.read_csv(SPLIT_DIR / "openi_test_labeled.csv")
df_test  = df_test[df_test["id"].isin(test_ids)].reset_index(drop=True)

df_train = pd.read_csv(SPLIT_DIR / "openi_train_labeled.csv")
df_train = df_train[df_train["id"].isin(train_ids)].reset_index(drop=True)

# Ensure the label columns are consistent with disease and normal groups
label_cols = list(disease_groups.keys()) + list(normal_groups.keys())
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
print("Min relevant items per query:", min(rel_counts))
print("Max relevant items per query:", max(rel_counts))
print("Avg relevant items per query:", sum(rel_counts)/len(rel_counts))

with open(GT_DIR / "test_relevance.json", "w") as f:
    json.dump(test_relevance, f, indent=2)

with open(GT_DIR / "test_to_train_relevance.json", "w") as f:
    json.dump(test_to_train, f, indent=2)

print(f"Saved generalization GT -> {GT_DIR/'test_relevance.json'}")
print(f"Saved historical  GT -> {GT_DIR/'test_to_train_relevance.json'}")
