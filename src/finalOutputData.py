import pandas as pd
import ast
from labeledData import disease_groups, normal_groups
from pathlib import Path

# Resolve paths...
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

vdata_path = BASE_DIR / "outputs" / "openi_labels_verified_final.csv"
out_path   = BASE_DIR / "outputs" / "openi_labels_final.csv"

# Build your one‑hot label columns
combined   = {**disease_groups, **normal_groups}
label_cols = list(combined.keys())

# Load the CSV
df = pd.read_csv(vdata_path)

# If there's a duplicate ID column like "id.1", drop it
for dup in [c for c in df.columns if c.startswith("id.")]:
    df = df.drop(columns=dup)

def safe_parse_list(val):
    """Parse a Python‑style or JSON list literal; return [] on error/NaN."""
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        parsed = ast.literal_eval(val)
        return [str(x).strip() for x in parsed] if isinstance(parsed, (list, tuple)) else []
    except Exception:
        return []

# Make sure all label columns are ints to start
df[label_cols] = df[label_cols].fillna(0).astype(int)

# Now apply remove/add via an explicit loop
for idx, row in df.iterrows():
    to_remove = safe_parse_list(row.get("remove", "[]"))
    to_add    = safe_parse_list(row.get("add",    "[]"))

    # Remove any labels
    for lbl in to_remove:
        if lbl in label_cols:
            df.at[idx, lbl] = 0

    # Add any labels
    for lbl in to_add:
        if lbl in label_cols:
            df.at[idx, lbl] = 1

# Collect final labels into a list
df["final_labels"] = df[label_cols].apply(
    lambda r: [lbl for lbl, v in r.items() if v == 1],
    axis=1
)

# Save
df.to_csv(out_path, index=False)
print(f"Wrote cleaned labels to {out_path}")
