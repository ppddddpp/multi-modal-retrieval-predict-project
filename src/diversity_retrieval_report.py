import json
from pathlib import Path
import numpy as np
import torch
from helper import (
    compare_maps,
    to_numpy,
    safe_unpack_topk,
    heatmap_to_base64_overlay,
    resize_to_match,
    save_b64_map,
    find_dicom_file,
    load_report_lookup_via_parser,
    model,
    preproc,
    tokenizer,
    cfg,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs" / "retrieval_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR = OUT_DIR / "overlays"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

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
    # fallback if model has no parameters (safe default)
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
                # last resort: try numpy()
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

        # Accept attention keys and convert to numpy immediately 
        att_maps_q = _to_numpy_map(out.get("attention_map", {}) or {})
        ig_maps_q = out.get("ig_maps", {}) or {}
        ig_maps_q = {int(k): to_numpy(v) for k, v in ig_maps_q.items()} if ig_maps_q else {}
        gradcam_maps_q = out.get("gradcam_maps", {}) or {}
        gradcam_maps_q = {int(k): to_numpy(v) for k, v in gradcam_maps_q.items()} if gradcam_maps_q else {}

        # quick debug listing
        present_att_keys = [k for k in ATT_VARIANTS if k in att_maps_q and att_maps_q[k] is not None]
        if not present_att_keys:
            print(f"[WARN] Query {qid}: no usable attention maps; keys found: {list(att_maps_q.keys())}")
        else:
            print(f"[DEBUG] Query {qid}: attention keys: {present_att_keys}")

        # determine main_target robustly (prefer topk[0], then first ig)
        main_target = None
        topk_idx = safe_unpack_topk(out.get("topk_idx", []))
        if topk_idx:
            try:
                # safe_unpack_topk expected to return a list-like
                main_target = int(topk_idx[0])
            except Exception:
                main_target = None
        if main_target is None and len(ig_maps_q) > 0:
            main_target = list(ig_maps_q.keys())[0]

        # prepare img_vis for overlays (normalize to 0-255 uint8)
        q_arr = q_tensor.squeeze().cpu().numpy().astype(np.float32)
        arr_max, arr_min = np.nanmax(q_arr), np.nanmin(q_arr)
        if np.isfinite(arr_max - arr_min) and (arr_max - arr_min) > 0:
            img_vis = ((q_arr - arr_min) / (arr_max - arr_min)) * 255.0
        else:
            img_vis = np.clip(q_arr, 0, 1) * 255.0
        img_vis = img_vis.astype(np.uint8)

        # Save query overlays (guard each item)
        def _maybe_overlay_and_save(img, amap, qid, rid, tag):
            if amap is None:
                return None
            try:
                b64 = heatmap_to_base64_overlay(img, amap, alpha=0.45)
                return save_b64_map(b64, qid, rid, tag)
            except Exception as e:
                print(f"[WARN] overlay save failed for {qid}/{rid}/{tag}: {e}")
                return None

        attn_q_txt_path = _maybe_overlay_and_save(img_vis, att_maps_q.get("txt2img"), qid, None, "attn_txt")
        attn_q_img_path = _maybe_overlay_and_save(img_vis, att_maps_q.get("img2txt"), qid, None, "attn_img")

        # pick the first available combination-attention variant
        comb_map = None
        for k in ("comb", "comb_img", "comb_txt", "att_comb_tensor", "final_patch_map"):
            if att_maps_q.get(k) is not None:
                comb_map = att_maps_q.get(k)
                break
        attn_q_comb_path = _maybe_overlay_and_save(img_vis, comb_map, qid, None, "attn_comb")

        query_ig_path = _maybe_overlay_and_save(img_vis, ig_maps_q.get(main_target) if main_target is not None else None, qid, None, "ig")
        query_gc_path = _maybe_overlay_and_save(img_vis, gradcam_maps_q.get(main_target) if main_target is not None else None, qid, None, "gradcam")

        # retrieval results from predict() output
        retrieval_ids = out.get("retrieval_ids", [])[:TOPK]
        retrieval_dists = out.get("retrieval_dists", [])[:TOPK]

        count += 1
        print(f"Query: {qid}, Target: {main_target}, Current: {count}/{len(query_ids)}")

        per_retrieved = []
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

                # call model.get_explain_score for the retrieved item and use its returned dict
                out_ret = model.get_explain_score(r_tensor, txt_ids, txt_mask, main_target)

                # normalize retrieved maps to numpy
                att_maps_r = _to_numpy_map(out_ret.get("attention_map", {}) or {})
                ig_maps_r = out_ret.get("ig_maps", {}) or {}
                ig_maps_r = {int(k): to_numpy(v) for k, v in ig_maps_r.items()} if ig_maps_r else {}
                gradcam_maps_r = out_ret.get("gradcam_maps", {}) or {}
                gradcam_maps_r = {int(k): to_numpy(v) for k, v in gradcam_maps_r.items()} if gradcam_maps_r else {}

                # prepare visualization array for this retrieved image
                img_arr = r_tensor.squeeze().cpu().numpy().astype(np.float32)
                arr_max, arr_min = np.nanmax(img_arr), np.nanmin(img_arr)
                if np.isfinite(arr_max - arr_min) and (arr_max - arr_min) > 0:
                    img_vis_r = ((img_arr - arr_min) / (arr_max - arr_min)) * 255.0
                else:
                    img_vis_r = np.clip(img_arr, 0, 1) * 255.0
                img_vis_r = img_vis_r.astype(np.uint8)

                # Save retrieved overlays (guard for multiple naming schemes)
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

                # compute comparison metrics (only when maps exist and shapes match)
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
                                # Resize qmap -> rmap shape (you could also resize both to a canonical size)
                                qmap_resized = resize_to_match(qmap, rmap)

                                # Normalize
                                def _norm01(x):
                                    xm, xM = np.nanmin(x), np.nanmax(x)
                                    if np.isfinite(xM - xm) and (xM - xm) > 0:
                                        return (x - xm) / (xM - xm)
                                    else:
                                        return x - xm  # all zeros

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
                                # If resize or compare fails, warn and skip that metric
                                print(f"[WARN] compare_maps resize/compare failed for {qid}/{rid}/{key}: {e}")
                                # Optionally store the failure
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

            except Exception as e:
                rec["error"] = str(e)

            per_retrieved.append(rec)

        results.append({"qid": qid, "query_report": q_report, "retrieval": per_retrieved})

    except Exception as e:
        print(f"[WARN] query {qid} failed: {e}")

# save the JSON
with open(OUT_DIR / "retrieval_report.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved report:", OUT_DIR / "retrieval_report.json")
