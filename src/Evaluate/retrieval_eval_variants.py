from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import json, numpy as np, torch, pandas as pd, time
from tqdm import tqdm
from typing import List
from Helpers import Config
from Model import MultiModalRetrievalModel
from Retrieval import make_retrieval_engine
from Retrieval.reranker import Reranker
from DataHandler import parse_openi_xml, build_dataloader
from Helpers.retrieval_metrics import precision_at_k, mean_average_precision, mean_reciprocal_rank,recall_at_k, ndcg_at_k
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
EMBED_DIR = BASE_DIR / "embeddings"
GT_DIR = BASE_DIR / "ground_truths"
SPLIT_DIR = BASE_DIR / "splited_data"
CKPT_PATH = BASE_DIR / "checkpoints" / "model_best.pt"
MODEL_DIR = BASE_DIR / "models"
KG_DIR = BASE_DIR / "knowledge_graph"
XML_DIR = BASE_DIR / "data" / "openi" / "xml" / "NLMCXR_reports" / "ecgen-radiology"
DICOM_ROOT = BASE_DIR / "data" / "openi" / "dicom"
LABELS_CSV = BASE_DIR / "outputs" / "openi_labels_final.csv"

def evaluate_variant(variant_name, reranker=None, k=5):
    print(f"\n[INFO] === Evaluating Variant: {variant_name} ===")

    cfg = Config.load(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    with open(GT_DIR / "test_relevance.json") as f:
        gt_general = json.load(f)
    with open(GT_DIR / "test_to_train_relevance.json") as f:
        gt_historical = json.load(f)

    engine_testdb = make_retrieval_engine(
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

    combined_groups = {**disease_groups, **normal_groups, **finding_groups, **symptom_groups}
    label_cols = list(combined_groups.keys())
    parsed_records = parse_openi_xml(XML_DIR, DICOM_ROOT, combined_groups=combined_groups)
    labels_df = pd.read_csv(LABELS_CSV).set_index("id")

    with open(SPLIT_DIR / "test_split_ids.json") as f:
        test_ids = json.load(f)
    records = [r for r in parsed_records if str(r["id"]) in test_ids]

    test_loader = build_dataloader(records, batch_size=cfg.batch_size, mean=0.5, std=0.25, shuffle=False)

    model = MultiModalRetrievalModel(
        joint_dim=cfg.joint_dim,
        num_heads=cfg.num_heads,
        num_fusion_layers=cfg.num_fusion_layers,
        num_classes=len(label_cols),
        fusion_type=cfg.fusion_type,
        swin_ckpt_path=MODEL_DIR / "swin_checkpoint.safetensors",
        bert_local_dir=MODEL_DIR / "clinicalbert_local",
        checkpoint_path=str(CKPT_PATH),
        use_shared_ffn=cfg.use_shared_ffn,
        use_cls_only=cfg.use_cls_only,
        device=device,
        training=False
    ).to(device)
    model.eval()

    all_ret_gen, all_rel_gen, gen_times = [], [], []
    all_ret_hist, all_rel_hist, hist_times = [], [], []

    for batch in tqdm(test_loader, desc=f"Retrieval ({variant_name})"):
        img = batch["image"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attn_mask"].to(device)
        qids = batch["id"]

        with torch.no_grad():
            outputs = model(img, ids, mask, return_attention=False)
            joint_embs = outputs["joint_emb"].cpu().numpy()

        for i, qid in enumerate(qids):
            q_vec = joint_embs[i]
            t0 = time.perf_counter()
            ret_ids, _ = engine_testdb.retrieve(q_vec, K=k, reranker=reranker, query_id=qid, seed=cfg.seed)
            gen_times.append(time.perf_counter() - t0)
            all_ret_gen.append(ret_ids)
            all_rel_gen.append(gt_general[qid])

            t0 = time.perf_counter()
            ret_ids, _ = engine_traindb.retrieve(q_vec, K=k, reranker=reranker, query_id=qid, seed=cfg.seed)
            hist_times.append(time.perf_counter() - t0)
            all_ret_hist.append(ret_ids)
            all_rel_hist.append(gt_historical[qid])

    def summarize(name, rets, rels, times):
        p = np.mean([precision_at_k(r, rel, k=k) for r, rel in zip(rets, rels)])
        r = np.mean([recall_at_k(r, rel, k=k) for r, rel in zip(rets, rels)])
        mAP = mean_average_precision(rets, rels, k=k)
        mrr = mean_reciprocal_rank(rets, rels)
        ndcg = np.mean([ndcg_at_k(r, rel, k=k) for r, rel in zip(rets, rels)])
        avg_time = 1000 * np.mean(times)
        print(f"{name}  P@{k}: {p:.4f}, R@{k}: {r:.4f}, mAP: {mAP:.4f}, MRR: {mrr:.4f}, nDCG@{k}: {ndcg:.4f}, AvgTime: {avg_time:.2f} ms")
        return dict(P=p, R=r, mAP=mAP, MRR=mrr, nDCG=ndcg, Time_ms=avg_time)

    result = {
        "variant": variant_name,
        "general": summarize("Generalization", all_ret_gen, all_rel_gen, gen_times),
        "historical": summarize("Historical", all_ret_hist, all_rel_hist, hist_times)
    }
    return result

if __name__ == "__main__":
    results = {}
    # --- Baseline (no reranker)
    results["baseline"] = evaluate_variant("baseline", reranker=None, k=5)

    # --- KG only
    results["kg_only"] = evaluate_variant("kg_only", reranker=Reranker(KG_DIR, LABELS_CSV, alpha=0.0, beta=0.0, gamma=1.0), k=5)

    # --- Label Attention only
    results["la_only"] = evaluate_variant("la_only", reranker=Reranker(KG_DIR, LABELS_CSV, alpha=0.0, beta=1.0, gamma=0.0), k=5)

    # --- Hybrid (KG + LA)
    results["kg_la"] = evaluate_variant("kg_la", reranker=Reranker(KG_DIR, LABELS_CSV, alpha=0.4, beta=0.2, gamma=0.2), k=5)

    result_dir = BASE_DIR / "retrieval_eval_result"
    result_dir.mkdir(exist_ok=True)
    summary_path = result_dir / "summary_all_variants.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll evaluations completed. Summary saved to: {summary_path}")
