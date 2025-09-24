from pathlib import Path
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
from Evaluate import dataPhraseCheck
from Evaluate import edaLabeledCheck
from DataHandler import label2CSV, run_gemini_label_verifier, get_final_ouput_data, train_val_test_split
from Evaluate import get_eda_before_split, get_eda_after_split

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent
XML_DIR    = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
MODEL_PLACE = BASE_DIR / "models"
EDA_DIR = BASE_DIR / 'eda_data'
CHECK_RUN_DIR = BASE_DIR / "check_run"
OUTPUT_DIR = BASE_DIR / 'outputs'
SPLIT_DIR = BASE_DIR / 'splited_data'
EDA_DIR.mkdir(parents=True, exist_ok=True)
CHECK_RUN_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    combined_groups = {
        **disease_groups,
        **finding_groups,
        **symptom_groups,
        **normal_groups
    }
    dataPhraseCheck(xml_path=XML_DIR, dicom_path=DICOM_ROOT, 
                    model_path=MODEL_PLACE, check_run_dir=CHECK_RUN_DIR, 
                    combined_groups=combined_groups)
    
    edaLabeledCheck(xml_dir=XML_DIR, 
                    save_dir=EDA_DIR)
    
    label2CSV(xml_dir=XML_DIR, 
                dicom_dir=DICOM_ROOT, 
                out_path=OUTPUT_DIR / "openi_labels.csv", 
                combined_groups=combined_groups)

    """
    run_gemini_label_verifier(csv_in_path=OUTPUT_DIR / "openi_labels.csv", 
                                csv_out_path=OUTPUT_DIR / "openi_labels_verified.csv", 
                                batch_size=5, combined_groups=combined_groups)
    """

    get_final_ouput_data(validated_data_path=OUTPUT_DIR / "openi_labels_verified.csv", 
                            out_path=OUTPUT_DIR / "openi_labels_final.csv", 
                            combined=combined_groups)

    get_eda_before_split(xml_dir=XML_DIR, dicom_root=DICOM_ROOT, 
                        eda_dir=EDA_DIR, output_file=OUTPUT_DIR / "openi_labels_final.csv", 
                        combine_groups=combined_groups, drop_zero=True, 
                        save_cleaned=True, max_show=10, 
                        output_drop_zero=OUTPUT_DIR)
    
    train_val_test_split(xml_dir=XML_DIR, dicom_dir=DICOM_ROOT, 
                        combined_groups=combined_groups,label_csv=OUTPUT_DIR / "openi_labels_final_cleaned.csv",
                        split_dir=SPLIT_DIR, seed=2709, split_ratio=[0.8, 0.1, 0.1])
    
    get_eda_after_split(xml_dir=XML_DIR, dicom_root=DICOM_ROOT,split_dir=SPLIT_DIR, 
                        label_csv=OUTPUT_DIR / "openi_labels_final_cleaned.csv",combined_groups=combined_groups)