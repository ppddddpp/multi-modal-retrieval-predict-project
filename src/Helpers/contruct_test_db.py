import numpy as np, json, torch
import pandas as pd
from pathlib import Path
from .config import Config
from DataHandler import build_dataloader, parse_openi_xml

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

def construct_db_test(config_path=None, ckpt_path=None, 
                        split_dir=None, model_dir=None, 
                        xml_dir=None, dicom_root=None, 
                        embed_dir=None,
                        combined_groups=None):
    """
    Constructs a test database for the web application by sampling N records from the test split IDs.

    Parameters
    ----------
    config_path : str
        Path to config file
    ckpt_path : str
        Path to model checkpoint
    split_dir : str
        Path to folder containing train, validation, and test split IDs
    model_dir : str
        Path to folder containing model weights
    xml_dir : str
        Path to folder containing individual .xml report files
    dicom_root : str
        Root folder where .dcm files live (possibly nested)
    embed_dir : str
        Path to save the test embeddings to
    combined_groups : dict of str to list of str
        Dictionary where keys are disease/normal group names and values are lists of labels

    Returns
    -------
    None

    Saves a JSON file with the following format: {<rid>: {<dicom_path>, <report_text>, <labels>}} to the output directory
    """
    # Resolve paths
    if config_path is None:
        config_path = CONFIG_PATH
    if ckpt_path is None:
        ckpt_path = CKPT_PATH
    if split_dir is None:
        split_dir = SPLIT_DIR
    if xml_dir is None:
        xml_dir = XML_DIR
    if dicom_root is None:
        dicom_root = DICOM_ROOT
    if model_dir is None:
        model_dir = MODEL_DIR
    if embed_dir is None:
        embed_dir = EMBED_DIR
    if combined_groups is not None:
        raise ValueError("Please provide a least a list of disease groups and normal groups to label the report with.")
    
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
            meta_id = batch["id"][0]

            joint_emb, _, _ = model(img, ids, mask, return_attention=False)
            all_embs.append(joint_emb.cpu().numpy().squeeze(0))
            all_ids.append(meta_id)

    all_embs = np.vstack(all_embs)
    np.save(embed_dir / "test_joint_embeddings.npy", all_embs)

    with open(embed_dir / "test_ids.json", "w") as f:
        json.dump(all_ids, f)

    print(f"Saved test embeddings to {embed_dir/'test_embeddings.npy'}")
    print(f"Saved test IDs to {embed_dir/'test_ids.json'}")

if __name__ == "__main__":
    construct_db_test()
