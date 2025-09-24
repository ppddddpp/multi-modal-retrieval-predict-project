import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from Helpers import (
    compare_maps,
    to_numpy,
    safe_unpack_topk,
    heatmap_to_base64_overlay,
    resize_to_match,
    save_b64_map,
    find_dicom_file,
    load_report_lookup_via_parser,
)
from Retrieval.retrieval import make_retrieval_engine
from Helpers import Config
from Model import MultiModalRetrievalModel
from DataHandler import DICOMImagePreprocessor
from LabelData import normal_groups, disease_groups, finding_groups, symptom_groups

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "retrieval_diversity_score" / "retrieval_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR = OUT_DIR / "overlays"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = BASE_DIR / "config"
CKPT_PATH = BASE_DIR / "checkpoints" / "model_best.pt"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
MODEL_DIR = BASE_DIR / "models"
XML_DIR = BASE_DIR / "data" / "openi" / "xml" / "NLMCXR_reports" / "ecgen-radiology"
DICOM_ROOT = BASE_DIR / "data" / "openi" / "dicom"
OUTPUT_DIR = BASE_DIR / "outputs"
LABEL_CSV = BASE_DIR / "outputs" / "openi_labels_final.csv"

cfg = Config.load(CONFIG_DIR / "config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[loader] device = {device}")

# report lookup
report_lookup = load_report_lookup_via_parser(XML_DIR, DICOM_ROOT)

# retrieval engine
embeddings_feat = EMBEDDINGS_DIR / "train_joint_embeddings.npy"
embeddings_ids  = EMBEDDINGS_DIR / "train_ids.json"
if not embeddings_feat.exists() or not embeddings_ids.exists():
    raise FileNotFoundError(f"Embeddings or ids not found at {embeddings_feat} / {embeddings_ids}")

retriever = make_retrieval_engine(
    features_path=str(embeddings_feat),
    ids_path=str(embeddings_ids),
    method="dls",
    link_threshold=0.5,
    max_links=10
)

# image preprocessor & tokenizer
preproc = DICOMImagePreprocessor()
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", cache_dir=str(MODEL_DIR / "clinicalbert"))
combined_groups = {
    **disease_groups,
    **finding_groups,
    **symptom_groups,
    **normal_groups
    }
label_cols = sorted(combined_groups.keys())

model = MultiModalRetrievalModel(
    joint_dim=cfg.joint_dim,
    num_heads=cfg.num_heads,
    num_fusion_layers=cfg.num_fusion_layers,
    num_classes=len(label_cols),
    fusion_type=cfg.fusion_type,
    swin_ckpt_path=MODEL_DIR / "swin_checkpoint.safetensors",
    bert_local_dir= MODEL_DIR / "clinicalbert_local",
    checkpoint_path=str(CKPT_PATH),
    use_shared_ffn=cfg.use_shared_ffn,
    use_cls_only=cfg.use_cls_only,
    device=device,
    retriever=retriever
).to(device)
model.eval()

# config
TOPK = 5
TOPK_FRACS = [0.05, 0.20]  # for IoU comparisons

# Which query ids to use? e.g., test split
with open(BASE_DIR / "splited_data" / "test_split_ids.json") as f:
    query_ids = json.load(f)

# (re)load report lookup to ensure availability
report_lookup = load_report_lookup_via_parser(
    BASE_DIR / "data" / "openi" / "xml" / "NLMCXR_reports" / "ecgen-radiology",
    BASE_DIR / "data" / "openi" / "dicom",
)

results = []
device = None
try:
    device = next(model.parameters()).device
except Exception:
    device = torch.device("cpu")
model.eval()

# canonical attention keys
ATT_VARIANTS = [
    "txt2img", "img2txt",
    "comb", "comb_img", "comb_txt",
    "final_patch_map", "final_token_map",
    "att_txt_tensor", "att_img_tensor", "att_comb_tensor"
]

def _to_numpy_map(d: dict):
    """Convert all tensor-like values in dict to numpy arrays (leaves None alone)."""
    if not d:
        return {}
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
        else:
            try:
                out[k] = to_numpy(v)
            except Exception:
                try:
                    out[k] = np.array(v)
                except Exception:
                    out[k] = None
    return out

count = 0
for qid in query_ids:
    try:
        # get query report and dcm
        q_report = report_lookup.get(qid, "")
        if q_report == "":
            print(f"[WARN] query {qid} no report, skipping")
            continue

        q_dcm_path = find_dicom_file(qid)
        if q_dcm_path is None:
            print(f"[WARN] query {qid} missing DICOM, skipping")
            continue

        q_tensor = preproc(q_dcm_path).unsqueeze(0).to(device)

        # tokenize query text
        tokens = tokenizer(
            q_report or "",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.text_dim,
        )
        txt_ids = tokens.input_ids.to(device)
        txt_mask = tokens.attention_mask.to(device)

        # call model.predict (ask for explanations)
        out = model.predict(q_tensor, txt_ids, txt_mask, K=TOPK, explain=True)

        # query maps
        att_maps_q = _to_numpy_map(out.get("attention_map", {}) or {})
        ig_maps_q = out.get("ig_maps", {}) or {}
        ig_maps_q = {int(k): to_numpy(v) for k, v in ig_maps_q.items()} if ig_maps_q else {}
        gradcam_maps_q = out.get("gradcam_maps", {}) or {}
        gradcam_maps_q = {int(k): to_numpy(v) for k, v in gradcam_maps_q.items()} if gradcam_maps_q else {}

        # quick debug
        present_att_keys = [k for k in ATT_VARIANTS if k in att_maps_q and att_maps_q[k] is not None]
        if not present_att_keys:
            print(f"[WARN] Query {qid}: no usable attention maps; keys found: {list(att_maps_q.keys())}")
        else:
            print(f"[DEBUG] Query {qid}: attention keys: {present_att_keys}")

        # determine main target
        main_target = None
        topk_idx = safe_unpack_topk(out.get("topk_idx", []))
        if topk_idx:
            try:
                main_target = int(topk_idx[0])
            except Exception:
                main_target = None
        if main_target is None and len(ig_maps_q) > 0:
            main_target = list(ig_maps_q.keys())[0]

        # prepare query image vis
        q_arr = q_tensor.squeeze().cpu().numpy().astype(np.float32)
        arr_max, arr_min = np.nanmax(q_arr), np.nanmin(q_arr)
        if np.isfinite(arr_max - arr_min) and (arr_max - arr_min) > 0:
            img_vis = ((q_arr - arr_min) / (arr_max - arr_min)) * 255.0
        else:
            img_vis = np.clip(q_arr, 0, 1) * 255.0
        img_vis = img_vis.astype(np.uint8)

        # helper to save overlay
        def _maybe_overlay_and_save(img, amap, qid, rid, tag):
            if amap is None:
                return None
            try:
                b64 = heatmap_to_base64_overlay(img, amap, alpha=0.45)
                return save_b64_map(b64, qid, rid, tag)
            except Exception as e:
                print(f"[WARN] overlay save failed for {qid}/{rid}/{tag}: {e}")
                return None

        # save query overlays
        attn_q_txt_path = _maybe_overlay_and_save(img_vis, att_maps_q.get("txt2img"), qid, None, "attn_txt")
        attn_q_img_path = _maybe_overlay_and_save(img_vis, att_maps_q.get("img2txt"), qid, None, "attn_img")

        comb_map = None
        for k in ("comb", "comb_img", "comb_txt", "att_comb_tensor", "final_patch_map"):
            if att_maps_q.get(k) is not None:
                comb_map = att_maps_q.get(k)
                break
        attn_q_comb_path = _maybe_overlay_and_save(img_vis, comb_map, qid, None, "attn_comb")

        query_ig_path = _maybe_overlay_and_save(img_vis, ig_maps_q.get(main_target) if main_target is not None else None, qid, None, "ig")
        query_gc_path = _maybe_overlay_and_save(img_vis, gradcam_maps_q.get(main_target) if main_target is not None else None, qid, None, "gradcam")

        # retrieval results
        retrieval_ids = out.get("retrieval_ids", [])[:TOPK]
        retrieval_dists = out.get("retrieval_dists", [])[:TOPK]

        count += 1
        print(f"Query: {qid}, Target: {main_target}, Current: {count}/{len(query_ids)}")

        per_retrieved = []
        retrieved_final_patch_maps = []  # collect maps for retrieval→retrieval
        for rid, dist in zip(retrieval_ids, retrieval_dists):
            rec = {"qid": qid, "rid": rid, "dist": float(dist)}
            try:
                r_report = report_lookup.get(rid, "")
                r_dcm_path = find_dicom_file(rid)
                if r_dcm_path is None:
                    rec["error"] = "missing dcm"
                    per_retrieved.append(rec)
                    continue

                r_tensor = preproc(r_dcm_path).unsqueeze(0).to(device)

                out_ret = model.get_explain_score(r_tensor, txt_ids, txt_mask, main_target)

                att_maps_r = _to_numpy_map(out_ret.get("attention_map", {}) or {})
                ig_maps_r = out_ret.get("ig_maps", {}) or {}
                ig_maps_r = {int(k): to_numpy(v) for k, v in ig_maps_r.items()} if ig_maps_r else {}
                gradcam_maps_r = out_ret.get("gradcam_maps", {}) or {}
                gradcam_maps_r = {int(k): to_numpy(v) for k, v in gradcam_maps_r.items()} if gradcam_maps_r else {}

                img_arr = r_tensor.squeeze().cpu().numpy().astype(np.float32)
                arr_max, arr_min = np.nanmax(img_arr), np.nanmin(img_arr)
                if np.isfinite(arr_max - arr_min) and (arr_max - arr_min) > 0:
                    img_vis_r = ((img_arr - arr_min) / (arr_max - arr_min)) * 255.0
                else:
                    img_vis_r = np.clip(img_arr, 0, 1) * 255.0
                img_vis_r = img_vis_r.astype(np.uint8)

                attn_txt_path = _maybe_overlay_and_save(img_vis_r, att_maps_r.get("txt2img"), qid, rid, "attn_txt")
                attn_img_path = _maybe_overlay_and_save(img_vis_r, att_maps_r.get("img2txt"), qid, rid, "attn_img")

                comb_map_r = None
                for k in ("comb", "comb_img", "comb_txt", "att_comb_tensor", "final_patch_map"):
                    if att_maps_r.get(k) is not None:
                        comb_map_r = att_maps_r.get(k)
                        break
                attn_comb_path = _maybe_overlay_and_save(img_vis_r, comb_map_r, qid, rid, "attn_comb")

                ig_path = _maybe_overlay_and_save(img_vis_r, ig_maps_r.get(main_target) if main_target is not None else None, qid, rid, "ig")
                gc_path = _maybe_overlay_and_save(img_vis_r, gradcam_maps_r.get(main_target) if main_target is not None else None, qid, rid, "gradcam")

                # compute query→retrieval metrics
                cross = {}
                for key in ["txt2img", "img2txt", "comb", "comb_img", "comb_txt", "final_patch_map"]:
                    qmap = att_maps_q.get(key)
                    rmap = att_maps_r.get(key)
                    if qmap is not None and rmap is not None:
                        if qmap.shape == rmap.shape:
                            cm5 = compare_maps(qmap, rmap, topk_frac=TOPK_FRACS[0])
                            cm20 = compare_maps(qmap, rmap, topk_frac=TOPK_FRACS[1])
                            cross.update({
                                f"{key}_pearson": float(cm5.get("pearson", 0.0)),
                                f"{key}_spearman": float(cm5.get("spearman", 0.0)),
                                f"{key}_iou_5": float(cm5.get("iou_top5pct", 0.0)),
                                f"{key}_iou_20": float(cm20.get("iou_top20pct", 0.0)),
                            })
                        else:
                            try:
                                qmap_resized = resize_to_match(qmap, rmap)
                                # Normalize
                                def _norm01(x):
                                    xm, xM = np.nanmin(x), np.nanmax(x)
                                    if np.isfinite(xM - xm) and (xM - xm) > 0:
                                        return (x - xm) / (xM - xm)
                                    else:
                                        return x - xm
                                qn = _norm01(qmap_resized)
                                rn = _norm01(rmap.astype(np.float32))

                                cm5 = compare_maps(qn, rn, topk_frac=TOPK_FRACS[0])
                                cm20 = compare_maps(qn, rn, topk_frac=TOPK_FRACS[1])
                                cross.update({
                                    f"{key}_pearson": float(cm5.get("pearson", 0.0)),
                                    f"{key}_spearman": float(cm5.get("spearman", 0.0)),
                                    f"{key}_iou_5": float(cm5.get("iou_top5pct", 0.0)),
                                    f"{key}_iou_20": float(cm20.get("iou_top20pct", 0.0)),
                                })
                            except Exception as e:
                                print(f"[WARN] compare_maps resize/compare failed for {qid}/{rid}/{key}: {e}")
                                cross[f"{key}_error"] = str(e)

                # IG vs IG
                if ig_maps_q and ig_maps_r and main_target in ig_maps_q and main_target in ig_maps_r:
                    if ig_maps_q[main_target].shape == ig_maps_r[main_target].shape:
                        cm5 = compare_maps(ig_maps_q[main_target], ig_maps_r[main_target], topk_frac=TOPK_FRACS[0])
                        cm20 = compare_maps(ig_maps_q[main_target], ig_maps_r[main_target], topk_frac=TOPK_FRACS[1])
                        cross.update({
                            "ig_ig_pearson": float(cm5.get("pearson", 0.0)),
                            "ig_ig_spearman": float(cm5.get("spearman", 0.0)),
                            "ig_ig_iou_5": float(cm5.get("iou_top5pct", 0.0)),
                            "ig_ig_iou_20": float(cm20.get("iou_top20pct", 0.0)),
                        })

                # GradCAM vs GradCAM
                if gradcam_maps_q and gradcam_maps_r and main_target in gradcam_maps_q and main_target in gradcam_maps_r:
                    if gradcam_maps_q[main_target].shape == gradcam_maps_r[main_target].shape:
                        cm5 = compare_maps(gradcam_maps_q[main_target], gradcam_maps_r[main_target], topk_frac=TOPK_FRACS[0])
                        cm20 = compare_maps(gradcam_maps_q[main_target], gradcam_maps_r[main_target], topk_frac=TOPK_FRACS[1])
                        cross.update({
                            "gc_gc_pearson": float(cm5.get("pearson", 0.0)),
                            "gc_gc_spearman": float(cm5.get("spearman", 0.0)),
                            "gc_gc_iou_5": float(cm5.get("iou_top5pct", 0.0)),
                            "gc_gc_iou_20": float(cm20.get("iou_top20pct", 0.0)),
                        })

                rec.update({
                    "report": r_report,
                    "attn_txt_path": attn_txt_path,
                    "attn_img_path": attn_img_path,
                    "attn_comb_path": attn_comb_path,
                    "ig_path": ig_path,
                    "gradcam_path": gc_path,
                    "compare_metrics": cross,
                })

                # keep final_patch_map for retrieval to retrieval score
                if att_maps_r.get("final_patch_map") is not None:
                    retrieved_final_patch_maps.append(att_maps_r["final_patch_map"])

            except Exception as e:
                rec["error"] = str(e)

            per_retrieved.append(rec)

        # compute retrieval to retrieval diversity
        if len(retrieved_final_patch_maps) > 1:
            overlaps = []
            for i in range(len(retrieved_final_patch_maps)):
                for j in range(i + 1, len(retrieved_final_patch_maps)):
                    cm = compare_maps(
                        retrieved_final_patch_maps[i],
                        retrieved_final_patch_maps[j],
                        topk_frac=TOPK_FRACS[0],
                    )
                    overlaps.append(cm.get("iou_top5pct", 0.0))
            avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0
            avg_diversity = 1.0 - avg_overlap
        else:
            avg_overlap, avg_diversity = None, None

        results.append({
            "qid": qid,
            "query_report": q_report,
            "retrieval": per_retrieved,
            "retrieval_overlap_iou5": avg_overlap,
            "retrieval_diversity_score": avg_diversity,
        })

    except Exception as e:
        print(f"[WARN] query {qid} failed: {e}")

# save the JSON
with open(OUT_DIR / "retrieval_report.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved report:", OUT_DIR / "retrieval_report.json")
