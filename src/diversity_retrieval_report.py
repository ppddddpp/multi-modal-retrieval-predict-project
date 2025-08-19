import json
from pathlib import Path
import numpy as np
import torch
from helper import (
    compare_maps, 
    to_numpy,
    heatmap_to_base64_overlay,
    save_b64_map,
    find_dicom_file,
    load_report_lookup_via_parser,
    model,
    preproc,
    tokenizer,
    report_lookup,
    cfg
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
device = next(model.parameters()).device
model.eval()

# helper to robustly unwrap topk_idx candidates
def _unwrap_candidate(cand):
    """Recursively try to get a scalar out of lists/tuples/tensors.
    Returns int/float/str or None if cannot unwrap.
    """
    if cand is None:
        return None
    if torch.is_tensor(cand):
        try:
            return cand.item()
        except Exception:
            try:
                return cand.flatten()[0].item()
            except Exception:
                return None
    if isinstance(cand, (list, tuple)) and len(cand) > 0:
        return _unwrap_candidate(cand[0])
    if isinstance(cand, (int, float, str)):
        return cand
    return None

count = 0
for qid in query_ids:
    try:
        # get query report and dcm
        q_report = report_lookup.get(qid, "")
        if q_report == "":
            raise Exception("query report not found")

        q_dcm_path = find_dicom_file(qid)
        if q_dcm_path is None:
            print(f"[WARN] query {qid} missing DICOM, skipping")
            continue

        q_tensor = preproc(q_dcm_path).unsqueeze(0).to(device)

        # get retrieval result
        tokens = tokenizer(
            q_report or "",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.text_dim,
        )
        txt_ids = tokens.input_ids.to(device)
        txt_mask = tokens.attention_mask.to(device)

        out = model.predict(q_tensor, txt_ids, txt_mask, explain=True)

        retrieval_ids = out.get("retrieval_ids", [])[:TOPK]
        retrieval_dists = out.get("retrieval_dists", [])[:TOPK]

        att_maps_q = out.get("attention_map", {}) or {}
        if not all(k in att_maps_q for k in ("img2txt", "txt2img", "comb")):
            raise RuntimeError("Attention maps not found in output")
        else:
            print("attention maps found")

        # IG maps (target-based: class indices)
        ig_maps_q = out.get("ig_maps", {}) or {}
        ig_maps_q = {int(k): to_numpy(v) for k, v in ig_maps_q.items()}
        if not ig_maps_q:  # just check non-empty
            raise RuntimeError("IG maps not found in output")
        else:
            print(f"IG maps found for targets: {list(ig_maps_q.keys())}")

        # Grad-CAM maps (target-based: class indices)
        gradcam_maps_q = out.get("gradcam_maps", {}) or {}
        gradcam_maps_q = {int(k): to_numpy(v) for k, v in gradcam_maps_q.items()}
        if not gradcam_maps_q:
            raise RuntimeError("Grad-CAM maps not found in output")
        else:
            print(f"Grad-CAM maps found for targets: {list(gradcam_maps_q.keys())}")

        # Saving query maps
        if att_maps_q.get("img2txt") is None or att_maps_q.get("txt2img") is None or att_maps_q.get("comb") is None:
            raise RuntimeError("Attention maps not found in output")
        
        if ig_maps_q.get(main_target) is None:
            raise RuntimeError("IG maps not found in output")
        
        if gradcam_maps_q.get(main_target) is None:
            raise RuntimeError("Grad-CAM maps not found in output")

        attn_q_txt_path = save_b64_map(
            heatmap_to_base64_overlay(img_vis, att_maps_q.get("txt2img"), alpha=0.45)
            if att_maps_q.get("txt2img") is not None else None,
            qid, None, "attn_txt"
        )
        attn_q_img_path = save_b64_map(
            heatmap_to_base64_overlay(img_vis, att_maps_q.get("img2txt"), alpha=0.45)
            if att_maps_q.get("img2txt") is not None else None,
            qid, None, "attn_img"
        )
        attn_q_comb_path = save_b64_map(
            heatmap_to_base64_overlay(img_vis, att_maps_q.get("comb"), alpha=0.45)
            if att_maps_q.get("comb") is not None else None,
            qid, None, "attn_comb"
        )

        query_ig_path = save_b64_map(
            heatmap_to_base64_overlay(img_vis, ig_maps_q.get(main_target), alpha=0.45)
            if main_target in ig_maps_q else None,
            qid, None, "ig"
        )

        query_gc_path = save_b64_map(
            heatmap_to_base64_overlay(img_vis, gradcam_maps_q.get(main_target), alpha=0.45)
            if main_target in gradcam_maps_q else None,
            qid, None, "gradcam"
        )

        # resolve main target robustly
        main_target = None
        topk_idx = out.get("topk_idx", [])
        if topk_idx:
            candidate = _unwrap_candidate(topk_idx[0])
            if candidate is not None:
                try:
                    main_target = int(candidate)
                except Exception:
                    main_target = candidate

        if main_target is None and ig_maps_q:
            # fallback to first ig_maps_q key
            first_key = next(iter(ig_maps_q.keys()))
            try:
                main_target = int(first_key)
            except Exception:
                main_target = first_key

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

                att_maps_r, ig_maps_r, gradcam_maps_r = {}, {}, {}

                try:
                    out_ret = model.get_explain_score(
                        r_tensor.unsqueeze(0).to(device),
                        txt_ids, txt_mask,main_target
                    )

                    # Use same keys/structure as query
                    att_maps_r = out.get("attention_map", {}) or {}
                    if not all(k in att_maps_r for k in ("img2txt", "txt2img", "comb")):
                        raise RuntimeError("Attention maps not found in output")
                    else:
                        print("attention maps found")

                    # IG maps (target-based: class indices)
                    ig_maps_r = out.get("ig_maps", {}) or {}
                    ig_maps_r = {int(k): to_numpy(v) for k, v in ig_maps_r.items()}
                    if not ig_maps_r:  # just check non-empty
                        raise RuntimeError("IG maps not found in output")
                    else:
                        print(f"IG maps found for targets: {list(ig_maps_r.keys())}")

                    # Grad-CAM maps (target-based: class indices)
                    gradcam_maps_r = out.get("gradcam_maps", {}) or {}
                    gradcam_maps_r = {int(k): to_numpy(v) for k, v in gradcam_maps_r.items()}
                    if not gradcam_maps_r:
                        raise RuntimeError("Grad-CAM maps not found in output")
                    else:
                        print(f"Grad-CAM maps found for targets: {list(gradcam_maps_r.keys())}")

                except Exception as e:
                    print(f"[WARN] failed to compute explanations for retrieved {rid}: {e}")

                # visual overlays: make a 2D uint8 image for overlay helper
                img_arr = r_tensor.squeeze().cpu().numpy().astype(np.float32)
                arr_max, arr_min = np.nanmax(img_arr), np.nanmin(img_arr)
                range_ = arr_max - arr_min
                if np.isfinite(range_) and range_ > 0:
                    img_vis = ((img_arr - arr_min) / range_) * 255.0
                else:
                    img_vis = np.clip(img_arr, 0, 1) * 255.0
                img_vis = img_vis.astype(np.uint8)

                # Attention maps
                attn_txt_path = save_b64_map(
                    heatmap_to_base64_overlay(img_vis, att_maps_r.get("txt2img"), alpha=0.45)
                    if att_maps_r.get("txt2img") is not None else None,
                    qid, rid, "attn_txt"
                )

                attn_img_path = save_b64_map(
                    heatmap_to_base64_overlay(img_vis, att_maps_r.get("img2txt"), alpha=0.45)
                    if att_maps_r.get("img2txt") is not None else None,
                    qid, rid, "attn_img"
                )

                attn_comb_path = save_b64_map(
                    heatmap_to_base64_overlay(img_vis, att_maps_r.get("comb"), alpha=0.45)
                    if att_maps_r.get("comb") is not None else None,
                    qid, rid, "attn_comb"
                )

                # IG map
                ig_path = save_b64_map(
                    heatmap_to_base64_overlay(img_vis, ig_maps_r, alpha=0.45)
                    if ig_maps_r is not None else None,
                    qid, rid, "ig"
                )

                # GradCAM map
                gc_path = save_b64_map(
                    heatmap_to_base64_overlay(img_vis, gradcam_maps_r, alpha=0.45)
                    if gradcam_maps_r is not None else None,
                    qid, rid, "gradcam"
                )

                cross = {}
                # attention vs attention
                for key in ["txt2img", "img2txt", "comb"]:
                    qmap = att_maps_q.get(key)
                    rmap = att_maps_r.get(key)
                    if qmap is not None and rmap is not None and qmap.shape == rmap.shape:
                        cm5 = compare_maps(qmap, rmap, topk_frac=TOPK_FRACS[0])
                        cm20 = compare_maps(qmap, rmap, topk_frac=TOPK_FRACS[1])
                        cross.update({
                            f"{key}_pearson": cm5.get("pearson", 0.0),
                            f"{key}_spearman": cm5.get("spearman", 0.0),
                            f"{key}_iou_5": cm5.get("iou_top5pct", 0.0),
                            f"{key}_iou_20": cm20.get("iou_top20pct", 0.0),
                        })

                # IG vs IG
                if ig_maps_q and ig_maps_r is not None and main_target in ig_maps_q:
                    if ig_maps_q[main_target].shape == ig_maps_r.shape:
                        cm5 = compare_maps(ig_maps_q[main_target], ig_maps_r, topk_frac=TOPK_FRACS[0])
                        cm20 = compare_maps(ig_maps_q[main_target], ig_maps_r, topk_frac=TOPK_FRACS[1])
                        cross.update({
                            "ig_ig_pearson": cm5.get("pearson", 0.0),
                            "ig_ig_spearman": cm5.get("spearman", 0.0),
                            "ig_ig_iou_5": cm5.get("iou_top5pct", 0.0),
                            "ig_ig_iou_20": cm20.get("iou_top20pct", 0.0),
                        })

                # GradCAM vs GradCAM
                if gradcam_maps_q and gradcam_maps_r is not None and main_target in gradcam_maps_q:
                    if gradcam_maps_q[main_target].shape == gradcam_maps_r.shape:
                        cm5 = compare_maps(gradcam_maps_q[main_target], gradcam_maps_r, topk_frac=TOPK_FRACS[0])
                        cm20 = compare_maps(gradcam_maps_q[main_target], gradcam_maps_r, topk_frac=TOPK_FRACS[1])
                        cross.update({
                            "gc_gc_pearson": cm5.get("pearson", 0.0),
                            "gc_gc_spearman": cm5.get("spearman", 0.0),
                            "gc_gc_iou_5": cm5.get("iou_top5pct", 0.0),
                            "gc_gc_iou_20": cm20.get("iou_top20pct", 0.0),
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
