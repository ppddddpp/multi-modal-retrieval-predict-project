import numpy as np, json, torch
import pandas as pd
from pathlib import Path
from config import Config
from model import MultiModalRetrievalModel
from labeledData import disease_groups, normal_groups
from dataLoader import build_dataloader
from dataParser import parse_openi_xml

def main():
    BASE_DIR    = Path(__file__).resolve().parent.parent
    CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
    CKPT_PATH   = BASE_DIR / "checkpoints" / "model_best.pt"
    EMBED_DIR   = BASE_DIR / "embeddings"
    MODEL_DIR   = BASE_DIR / "models"
    SPLIT_DIR   = BASE_DIR / "splited_data"
    XML_DIR     = BASE_DIR / "data" / "openi" / "xml" / "NLMCXR_reports" / "ecgen-radiology"
    DICOM_ROOT  = BASE_DIR / "data" / "openi" / "dicom"
    EMBED_DIR.mkdir(exist_ok=True)

    cfg    = Config.load(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_cols = list(disease_groups.keys()) + list(normal_groups.keys())

    df_test = pd.read_csv(SPLIT_DIR / "openi_test_labeled.csv")
    with open(SPLIT_DIR / "test_split_ids.json") as f:
        test_ids = json.load(f)
    df_test = df_test[df_test["id"].isin(test_ids)].reset_index(drop=True)

    parsed_records = parse_openi_xml(XML_DIR, DICOM_ROOT)
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

    model = MultiModalRetrievalModel(
        joint_dim=cfg.joint_dim,
        num_heads=cfg.num_heads,
        num_classes=len(label_cols),
        fusion_type=cfg.fusion_type,
        swin_ckpt_path=MODEL_DIR / "swin_checkpoint.safetensors",
        bert_local_dir= MODEL_DIR / "clinicalbert_local",
        checkpoint_path=str(CKPT_PATH),
        use_shared_ffn=cfg.use_shared_ffn,
        use_cls_only=cfg.use_cls_only,
        device=device,
        training=True
    ).to(device)
    model.eval()

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
    np.save(EMBED_DIR / "test_joint_embeddings.npy", all_embs)

    with open(EMBED_DIR / "test_ids.json", "w") as f:
        json.dump(all_ids, f)

    print(f"Saved test embeddings to {EMBED_DIR/'test_embeddings.npy'}")
    print(f"Saved test IDs to {EMBED_DIR/'test_ids.json'}")

if __name__ == "__main__":
    main()
