from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import pandas as pd
import json
import random
import time
import logging
from google import genai
import os
from pathlib import Path
from dotenv import load_dotenv
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups

# Resolve paths
try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

XML_DIR    = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
MODEL_PLACE = BASE_DIR / "models"
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_PLACE)

class OpenIChecker:
    """
    Class to verify labels with Gemini
    """
    def __init__(self, gemini_api_keys):
        self.gemini_api_keys = gemini_api_keys

    def verify_labels_with_gemini(self, csv_path, output_path, batch_size=10, combined_groups=None):
        """
        Verify labels with Gemini.

        This function will take a CSV file of labels and verify them
        using the Gemini AI. It will output a new CSV file with the
        verified labels.

        Parameters
        ----------
        csv_path : str
            The path to the CSV file to verify.
        output_path : str
            The path to write the verified labels CSV file.
        batch_size : int, optional
            The number of labels to verify in each batch. Defaults to 10.
        combined_groups : dict, optional
            A dictionary of label names to lists of IDs to remove/add.

        Returns
        -------
        None

        Notes
        -----
        The function will write a merged CSV file to
        outputs/openi_labels_verified_final.csv, which includes all
        verified labels and retried Unknown rows.
        """
        df = pd.read_csv(csv_path)
        verified = []

        api_keys = self.gemini_api_keys.copy()
        print(f"Loaded {len(api_keys)} API keys")
        if len(api_keys) < 20:
            raise ValueError("Not enough API keys")

        random.shuffle(api_keys)

        for start in range(0, len(df), batch_size):
            time.sleep(5)
            batch = df.iloc[start : start + batch_size]
            prompt = self._build_verify_prompt(batch, combined_groups)

            response_text = None
            for api_key in api_keys:
                try:
                    client = genai.Client(api_key=api_key)
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    response_text = resp.text
                    break
                except Exception as e:
                    logging.error(f"[Verifier] Key {api_key[:4]} failed: {e}")
                    time.sleep(1)

            if response_text is None:
                for _ in batch.itertuples():
                    verified.append({
                        "llm_status":       "Unknown",
                        "llm_suggestions":  "{}"
                    })
            else:
                try:
                    if response_text.startswith("```json"):
                        response_text = response_text.strip()[7:-3].strip()
                    elif response_text.startswith("```"):
                        response_text = response_text.strip()[3:-3].strip()
                    results = json.loads(response_text)
                    print(f"Verifier returned {len(results)} results")
                except json.JSONDecodeError:
                    logging.error("Verifier returned invalid JSON; marking batch Unknown")
                    for _ in batch.itertuples():
                        verified.append({
                            "llm_status":       "Unknown",
                            "llm_suggestions":  "{}"
                        })
                else:
                    verified.extend(results)
            time.sleep(5)

        print(f"Verified {len(verified)} labels")
        verified_df = pd.DataFrame(verified)
        out = pd.concat([df.reset_index(drop=True), verified_df], axis=1)
        out.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Wrote verified labels to {output_path}")

    def _build_verify_prompt(self, batch_df, combined_groups=None):
        """
        Function for building a prompt for the LLM context to verify labels.

        Parameters
        ----------
        batch_df : pd.DataFrame
            A DataFrame of labels to verify.
        combined_groups : dict, optional
            A dictionary of label names to lists of IDs to remove/add.
        combined_groups : list, optional
            A list of label groups to combine for verification.
        Returns
        -------
        str
            A JSON string to be used as the prompt for the LLM context.

        Notes
        -----
        The function will raise a ValueError if no combined groups are provided.
        The function will provide label descriptions for LLM context by
        serializing the combined groups to a JSON string.
        The function will generate a prompt string by concatenating the
        instructions and the JSON string of the batch entries.
        """
        if combined_groups is None:
            raise ValueError("Please provide a least a list of disease groups and normal groups to label the report with.")
        
        # Provide label descriptions for LLM context
        label_info = json.dumps(combined_groups, indent=2)

        instructions = (
            "You are a clinical-label verifier. For each chest X‑ray report and its"
            " assigned one‑hot disease/normal labels, check whether the labels are:"
            "\n  • Correct: no changes needed"
            "\n  • Incorrect: remove any labels that don’t apply"
            "\n  • Missing: add any labels that were missed"
            "\nRespond *only* in JSON, as a list of objects with these keys:\n"
            "  {\n"
            "    \"id\": <the report’s id>,\n"
            "    \"llm_status\": \"Correct\"|\"Incorrect\"|\"Missing\",\n"
            "    \"remove\": [<labels to remove>],\n"
            "    \"add\":    [<labels to add>]\n"
            "  }\n"
            "\nHere are the label categories and their associated terms:\n"
            f"{label_info}\n"
            "\nNow review these records:\n\n"
        )

        entries = []
        for _, row in batch_df.iterrows():
            assigned = [c for c in batch_df.columns
                        if c not in ("id", "report_text")
                        and row[c] == 1]
            entries.append({
                "id":          row["id"],
                "report_text": row["report_text"],
                "labels":      assigned
            })

        return instructions + json.dumps(entries, indent=2)

    def _run_pass(self, csv_path, batch_size, output_path, combined_groups):
        """
        Internal helper to re-use verify_labels_with_gemini for retrying small batches.
        """
        self.verify_labels_with_gemini(csv_path=csv_path,
                                       output_path=output_path,
                                       batch_size=batch_size, 
                                       combined_groups=combined_groups)
        return pd.read_csv(output_path)

    def retry_unknowns(self, verified_csv_path, output_path, retry_batch_size=1, combined_groups=None):
        """
        Retry verifying labels on Unknown rows.

        Parameters
        ----------
        verified_csv_path : str
            The path to the CSV file of already verified labels.
        output_path : str
            The path to write the retried + merged results CSV file.
        retry_batch_size : int, optional
            The number of labels to retry in each batch. Defaults to 1.
        combined_groups : dict, optional
            A dictionary of label names to lists of IDs to remove/add.

        Returns
        -------
        pd.DataFrame
            The DataFrame of retried + merged results.

        Notes
        -----
        The function will write a merged CSV file to
        outputs/openi_labels_verified_final.csv, which includes all
        verified labels and retried Unknown rows.
        """
        # Load the already‑verified CSV
        verified_df = pd.read_csv(verified_csv_path)

        # Select only the Unknown rows
        unknown_df = verified_df[verified_df["llm_status"] == "Unknown"].copy()
        if unknown_df.empty:
            print("No Unknown rows to retry.")
            return verified_df

        print(f"Retrying {len(unknown_df)} Unknown rows…")

        # Run a single‑pass on just the unknowns
        temp_in  = verified_csv_path.parent / "temp_unknowns.csv"
        temp_out = verified_csv_path.parent / "temp_retry_results.csv"
        unknown_df.to_csv(temp_in, index=False)

        retry_df = self._run_pass(
            csv_path    = temp_in,
            batch_size  = retry_batch_size,
            output_path = temp_out,
            combined_groups=combined_groups
        )

        # Merge: drop old Unknowns, append new rows
        kept = verified_df[verified_df["llm_status"] != "Unknown"]
        merged = pd.concat([kept, retry_df], axis=0, ignore_index=True)

        #  Write final merged file
        merged.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Wrote retried + merged results to {output_path}")

        return merged

def run_gemini_label_verifier(csv_in_path=None, csv_out_path=None, batch_size=5, combined_groups=None):
    """
    Run the Gemini label verifier on a CSV file.

    This function loads Gemini API keys from a .env file,
    creates an OpenIChecker instance with those keys,
    runs the verifier on a CSV file, and then retries
    any Unknown rows with a single pass of Gemini.

    Parameters
    ----------
    csv_in_path : str, optional
        The path to the CSV file to verify. Defaults to
        outputs/openi_labels.csv.
    csv_out_path : str, optional
        The path to write the verified labels CSV file.
        Defaults to outputs/openi_labels_verified.csv.
    batch_size : int, optional
        The number of labels to verify in each batch. Defaults to 5.
    combined_groups : dict, optional
        A dictionary of label groups to combine for verification.

    Returns
    -------
    None

    Notes
    -----
    The final verified CSV file will be written to
    outputs/openi_labels_verified_final.csv, which includes all
    verified labels and retried Unknown rows.
    """
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    raw = os.getenv("GEMINI_KEYS", "")
    if not raw:
        raise ValueError("No GEMINI_KEYS found in .env file")
    gemini_api_keys = [k.strip() for k in raw.split(",") if k.strip()]

    checker = OpenIChecker(gemini_api_keys)

    csv_in  = BASE_DIR / "outputs" / "openi_labels.csv" if csv_in_path is None else csv_in_path
    csv_out = BASE_DIR / "outputs" / "openi_labels_verified.csv" if csv_out_path is None else csv_out_path

    checker.verify_labels_with_gemini(csv_path=csv_in,
                                        output_path=csv_out,
                                        batch_size=batch_size,
                                        combined_groups=combined_groups)

    csv_out_final = BASE_DIR / "outputs" / "openi_labels_verified_final.csv"

    checker.retry_unknowns(
    verified_csv_path = csv_out,
    output_path       = csv_out_final,
    retry_batch_size  = 1,
    combined_groups=combined_groups
    )
    
if __name__ == "__main__":
    combined_groups = {
        **disease_groups,
        **finding_groups,
        **symptom_groups,
        **normal_groups
    }
    run_gemini_label_verifier(combined_groups=combined_groups)