import json
import numpy as np
import torch
from pathlib import Path
import pandas as pd
from config import Config
from model import MultiModalRetrievalModel
from retrieval import make_retrieval_engine
from dataParser import parse_openi_xml
from retrieval_metrics import precision_at_k, recall_at_k
from dataLoader import build_dataloader
from labeledData import disease_groups, normal_groups

BASE_DIR    = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
EMBED_DIR   = BASE_DIR / "embeddings"
GT_DIR      = BASE_DIR / "ground_truths"
SPLIT_DIR   = BASE_DIR / "splited_data"
CKPT_PATH   = BASE_DIR.parent / "checkpoints" / "model_best.pt"
MODEL_DIR   = BASE_DIR / "models"
XML_DIR     = BASE_DIR / "data" / "openi" / "xml" / "NLMCXR_reports" / "ecgen-radiology"
DICOM_ROOT  = BASE_DIR / "data" / "openi" / "dicom"


def main(k=10):
    cfg = Config.load(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_cols = list(disease_groups.keys()) + list(normal_groups.keys())

    with open(GT_DIR / "test_relevance.json") as f:
        gt_general = json.load(f)

    with open(GT_DIR / "test_to_train_relevance.json") as f:
        gt_historical = json.load(f)

    engine_testdb  = make_retrieval_engine(
        str(EMBED_DIR / "test_joint_embeddings.npy"),
        str(EMBED_DIR / "test_ids.json"),
        method="dls",
        link_threshold=0.5,
        max_links=10
    )

    engine_traindb = make_retrieval_engine(
        str(EMBED_DIR / "train_joint_embeddings.npy"),
        str(EMBED_DIR / "train_ids.json"),
        method="dls",
        link_threshold=cfg.focal_ratio,
        max_links=int(cfg.num_heads)
    )

    model = MultiModalRetrievalModel(
        joint_dim=cfg.joint_dim,
        num_heads=cfg.num_heads,
        num_classes=len(label_cols),
        fusion_type=cfg.fusion_type,
        swin_ckpt_path=MODEL_DIR / "swin_checkpoint.safetensors",
        bert_local_dir= MODEL_DIR / "clinicalbert_local",
        checkpoint_path=str(CKPT_PATH),
        device=device,
        training=True
    ).to(device)
    model.eval()

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

    all_ret_gen,  all_rel_gen  = [], []
    all_ret_hist, all_rel_hist = [], []

    for batch in test_loader:
        qid = batch["id"][0]

        img   = batch["image"].to(device)
        ids   = batch["input_ids"].to(device)
        mask  = batch["attn_mask"].to(device)

        with torch.no_grad():
            joint_emb, _, _ = model(img, ids, mask, return_attention=False)

        q_emb = joint_emb.cpu().numpy()

        # Generalization: test to test
        ret_ids, _ = engine_testdb.retrieve(q_emb, K=5)
        all_ret_gen.append(ret_ids)
        all_rel_gen.append(gt_general[qid])

        # Historical: test to train
        ret_ids, _ = engine_traindb.retrieve(q_emb, K=5)
        all_ret_hist.append(ret_ids)
        all_rel_hist.append(gt_historical[qid])

    # Evaluate at K = 5
    p5_gen   = np.mean([precision_at_k(r, rel, k=k) for r, rel in zip(all_ret_gen, all_rel_gen)])
    r5_gen   = np.mean([recall_at_k(r, rel, k=k) for r, rel in zip(all_ret_gen, all_rel_gen)])

    p5_hist  = np.mean([precision_at_k(r, rel, k=k) for r, rel in zip(all_ret_hist, all_rel_hist)])
    r5_hist  = np.mean([recall_at_k(r, rel, k=k) for r, rel in zip(all_ret_hist, all_rel_hist)])

    print(f"Generalization (test to test)   P@{k}: {p5_gen:.4f},  R@{k}: {r5_gen:.4f}")
    print(f"Historical    (test to train)  P@{k}: {p5_hist:.4f}, R@{k}: {r5_hist:.4f}")

if __name__ == "__main__":
    main(k=5)