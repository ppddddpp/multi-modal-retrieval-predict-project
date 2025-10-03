from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import numpy as np, json, torch
import pandas as pd
from Helpers.config import Config
from DataHandler import build_dataloader, parse_openi_xml
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
CKPT_PATH   = BASE_DIR / "checkpoints" / "model_best.pt"
EMBED_DIR   = BASE_DIR / "embeddings"
MODEL_DIR   = BASE_DIR / "models"
SPLIT_DIR   = BASE_DIR / "splited_data"
XML_DIR     = BASE_DIR / "data" / "openi" / "xml" / "NLMCXR_reports" / "ecgen-radiology"
DICOM_ROOT  = BASE_DIR / "data" / "openi" / "dicom"
EMBED_DIR.mkdir(exist_ok=True)

def get_model(cfg, ckpt_path, model_dir, device, label_cols):
    if cfg is None:
        cfg = Config.load(CONFIG_PATH)
    
    from Model import MultiModalRetrievalModel
    model = MultiModalRetrievalModel(
        joint_dim=cfg.joint_dim,
        num_heads=cfg.num_heads,
        num_classes=len(label_cols),
        fusion_type=cfg.fusion_type,
        swin_ckpt_path=model_dir / "swin_checkpoint.safetensors",
        bert_local_dir= model_dir / "clinicalbert_local",
        checkpoint_path=str(ckpt_path),
        use_shared_ffn=cfg.use_shared_ffn,
        use_cls_only=cfg.use_cls_only,
        device=device,
        training=True
    ).to(device)
    model.eval()

    return model

def construct_db_test(config_path=CONFIG_PATH, ckpt_path=CKPT_PATH, 
                        split_dir=SPLIT_DIR, model_dir=MODEL_DIR, 
                        xml_dir=XML_DIR, dicom_root=DICOM_ROOT, 
                        embed_dir=EMBED_DIR,
                        combined_groups=None):
    """
    Construct a test database for the web application.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    ckpt_path : str
        Path to the model checkpoint.
    split_dir : str
        Path to the folder containing train, validation, and test split IDs.
    model_dir : str
        Path to the folder containing the model.
    xml_dir : str
        Path to the folder containing individual .xml report files.
    dicom_root : str
        Root folder where .dcm files live (possibly nested).
    embed_dir : str
        Path to the folder containing test embeddings and IDs.
    combined_groups : dict of str to list of str
        Dictionary where keys are disease/normal group names and values are lists of labels.

    Returns
    -------
    None

    Saves test embeddings to 'embed_dir/test_joint_embeddings.npy' and test IDs to 'embed_dir/test_ids.json'.
    """
    if combined_groups is not None:
        print("Using provided combined_groups")
    else:
        combined_groups = {**disease_groups, **normal_groups, **finding_groups, **symptom_groups}

    cfg    = Config.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_cols = list(combined_groups.keys())

    df_test = pd.read_csv(split_dir / "openi_test_labeled.csv")
    with open(split_dir / "test_split_ids.json") as f:
        test_ids = json.load(f)
    df_test = df_test[df_test["id"].isin(test_ids)].reset_index(drop=True)

    parsed_records = parse_openi_xml(xml_dir, dicom_root, combined_groups=combined_groups)
    labels_df = pd.read_csv(BASE_DIR / "outputs" / "openi_labels_final.csv").set_index("id")

    records = []
    for rec in parsed_records:
        rec_id = rec["id"]
        if rec_id in labels_df.index:
            label_vec = labels_df.loc[rec_id, label_cols].tolist()
            records.append({
                "id": rec["id"],
                "report_text": rec["report_text"],
                "dicom_path": rec["dicom_path"],
                "labels": label_vec
            })

    test_records = [r for r in records if r["id"] in test_ids]

    test_loader = build_dataloader(
        records=test_records,
        batch_size=cfg.batch_size,
        mean=0.5,
        std=0.25,
        shuffle=False
    )

    model = get_model(cfg, ckpt_path, model_dir, device, label_cols)

    all_embs = []
    all_ids  = []

    with torch.no_grad():
        for batch in test_loader:
            img  = batch["image"].to(device)
            ids  = batch["input_ids"].to(device)
            mask = batch["attn_mask"].to(device)

            out = model(img, ids, mask, return_attention=False)
            joint_emb = out["joint_emb"]          # shape: (B, D)
            logits    = out["logits"]             # shape: (B, num_classes)
            attn      = out["attn"]

            # Convert to numpy and store all embeddings in this batch
            all_embs.append(joint_emb.cpu().numpy())  
            all_ids.extend(batch["id"])           # extend works for list of IDs

    all_embs = np.vstack(all_embs)   # shape: (N, D)
    np.save(embed_dir / "test_joint_embeddings.npy", all_embs)

    with open(embed_dir / "test_ids.json", "w") as f:
        json.dump(all_ids, f)

    print(f"Saved test embeddings to {embed_dir/'test_joint_embeddings.npy'}")
    print(f"Saved test IDs to {embed_dir/'test_ids.json'}")

if __name__ == "__main__":
    construct_db_test()
