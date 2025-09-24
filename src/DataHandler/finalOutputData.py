from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import pandas as pd
import ast
from pathlib import Path
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
# Resolve paths...
try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

def get_final_ouput_data(validated_data_path = None, out_path = None, combined = None):
    """
    Apply manual label changes from a verified CSV to a final output CSV.

    Given a verified CSV, apply remove/add label changes and save the result to a final output CSV.

    Args:
        combined (dict): A dictionary of label names to lists of IDs to remove/add.

    Returns:
        None
    """
    vdata_path = BASE_DIR / "outputs" / "openi_labels_verified_final.csv" if validated_data_path is None else validated_data_path
    out_path   = BASE_DIR / "outputs" / "openi_labels_final.csv" if out_path is None else out_path

    if combined is None:
        raise Exception("Must provide a dictionary of label names to lists of IDs to remove/add.")

    # Build your one‑hot label columns
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

if __name__ == "__main__":
    combined_groups = {
        **disease_groups,
        **finding_groups,
        **symptom_groups,
        **normal_groups
    }
    get_final_ouput_data(combined=combined_groups)
