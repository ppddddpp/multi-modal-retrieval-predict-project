import io
import base64
import sys
import pickle
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from flask import Flask, render_template, request
import numpy as np
import torch
import time
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for rendering images
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
import torch.nn.functional as F

from Helpers import Config, compare_maps, heatmap_to_base64_overlay, attention_to_html
from DataHandler import DICOMImagePreprocessor, parse_openi_xml
from Model import MultiModalRetrievalModel
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
from Retrieval import make_retrieval_engine

plt.ioff()

# ── Project directories ────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
CKPT_DIR       = BASE_DIR / "checkpoints"
MODEL_DIR      = BASE_DIR / "models"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
OUTPUT_DIR    = BASE_DIR / "outputs"
XML_DIR = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
CONFIG_DIR     = BASE_DIR / "configs"
CKPT_PATH      = BASE_DIR / "checkpoints" / "model_best.pt"
REPORT_CACHE_PATH = OUTPUT_DIR / "openi_reports.pkl"
#────────────────────────────────────────────────────────────────────────────────

cfg = Config.load(CONFIG_DIR / "config.yaml")
app = Flask(__name__, static_folder="static", template_folder="templates")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_cols = list(disease_groups.keys()) + list(normal_groups.keys())
labels_df = pd.read_csv(OUTPUT_DIR / "openi_labels_final.csv").set_index("id")

def load_report_lookup_via_parser(xml_dir, dicom_root) -> dict:
    if REPORT_CACHE_PATH.exists():
        print(f"[INFO] Loading cached report lookup from {REPORT_CACHE_PATH.name}")
        with open(REPORT_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print("[INFO] Parsing reports using parse_openi_xml()...")
    parsed_records = parse_openi_xml(xml_dir, dicom_root)
    id_to_report = {
        rec["id"]: rec["report_text"]
        for rec in parsed_records
        if "id" in rec and "report_text" in rec
    }

    with open(REPORT_CACHE_PATH, "wb") as f:
        pickle.dump(id_to_report, f)
    print(f"[INFO] Cached {len(id_to_report)} reports to {REPORT_CACHE_PATH.name}")
    return id_to_report

def find_dicom_file(rid: str) -> Path:
    primary = list(DICOM_ROOT.rglob(f"{rid}.dcm"))
    if primary:
        return primary[0]

    # Try fallback without leading patient ID
    fallback_id = "_".join(rid.split("_")[1:])
    fallback = list(DICOM_ROOT.rglob(f"{fallback_id}.dcm"))
    if fallback:
        print(f"[INFO] Using fallback DICOM path: {fallback[0].name}")
        return fallback[0]

    raise FileNotFoundError(f"DICOM not found for either {rid}.dcm or {fallback_id}.dcm")

report_lookup = load_report_lookup_via_parser(XML_DIR, DICOM_ROOT)

retriever = make_retrieval_engine(
    features_path=str(EMBEDDINGS_DIR / "train_joint_embeddings.npy"),
    ids_path=str(EMBEDDINGS_DIR / "train_ids.json"),
    method="dls",
    link_threshold=0.5,
    max_links=10
)

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

preproc   = DICOMImagePreprocessor()
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def np_to_base64_img(arr: np.ndarray, cmap="gray") -> str:
    fig, ax = plt.subplots(figsize=(4, 4))

    # Handle (1, H, W) => (H, W)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        ax.imshow(arr, cmap=cmap)
    else:
        ax.imshow(arr)

    ax.axis("off")
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def resize_to_match(src_map, target_shape):
    """Resize src_map (H,W) to target_shape (H2,W2) using bilinear interpolation."""
    src_tensor = torch.tensor(src_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    resized = F.interpolate(src_tensor, size=target_shape, mode="bilinear", align_corners=False)
    return resized.squeeze().numpy()

def to_numpy(x):
    """Return a numpy array whether x is a torch tensor or numpy already."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def safe_unpack_topk(topk_any):
    """Normalize topk returned shapes to a flat python list."""
    if topk_any is None:
        return []
    # if nested lists (B x K) and user passed topk for batch, take first row
    if isinstance(topk_any, list) and len(topk_any) > 0 and isinstance(topk_any[0], (list, tuple)):
        return list(topk_any[0])
    # if numpy array
    try:
        return list(np.array(topk_any).tolist())
    except Exception:
        return list(topk_any)

@app.route("/", methods=["GET", "POST"])
def index():
    context = {}
    if request.method == "POST":
        print("[INFO] POST request")
        # --- Load inputs ---
        dcm_bytes  = request.files["dicom_file"].read()
        raw_tensor = preproc(dcm_bytes)                 # shape (C,H,W) or (1,H,W)
        img_tensor = raw_tensor.unsqueeze(0).to(device) # (1,C,H,W)
        threshold = float(request.form.get("threshold", 0.5))
        text_query = request.form.get("text_query", "") or ""
        tokens    = tokenizer(
            text_query,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.text_dim
        )
        txt_ids  = tokens.input_ids.to(device)
        txt_mask = tokens.attention_mask.to(device)
        
        # prepare orig_arr for visualization (H,W) scaled 0..1
        orig_arr = raw_tensor.squeeze().cpu().numpy()
        if orig_arr.dtype != np.float32 and orig_arr.max() > 1.0:
            orig_arr = orig_arr.astype(np.float32) / 255.0

        # --- Predict + Explain for query image ---
        start_time = time.perf_counter()
        out = model.predict(img_tensor, txt_ids, txt_mask, K=5, explain=True)
        predict_time = time.perf_counter() - start_time

        # Get explaining outputs
        att_maps_q = out.get("attention_map", {}) or {}
        valid_keys = ["txt2img", "img2txt", "comb_img", "comb_txt", "final_patch_map", "final_token_map"]
        
        if not any(k in att_maps_q and att_maps_q[k] is not None for k in valid_keys):
            raise RuntimeError("No usable attention maps found in output")
        else:
            print("Attention maps found:", [k for k in valid_keys if k in att_maps_q])

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

        # pick which IG target to show: prefer topk[0] if available
        topk_idx = safe_unpack_topk(out.get("topk_idx", []))
        topk_vals = safe_unpack_topk(out.get("topk_vals", []))
        main_target = None
        if topk_idx:
            try:
                main_target = int(topk_idx[0])
            except Exception:
                main_target = None
        if main_target is None and len(ig_maps_q) > 0:
            main_target = list(ig_maps_q.keys())[0]

        # prepare visuals for the query image
        context['img_orig'] = np_to_base64_img(img_tensor.squeeze().cpu().numpy(), cmap="gray")
        context["text_query"] = text_query
        context['attn_map_q_b64'] = None
        context['ig_map_q_b64']   = None
        context['gradcam_map_q_b64'] = None

        if att_maps_q is not None:
            try:
                # --- Attention overlays ---
                context['attn_txt2img_b64']   = heatmap_to_base64_overlay(orig_arr, att_maps_q.get("txt2img"), alpha=0.45)
                context['attn_comb_img_b64']  = heatmap_to_base64_overlay(orig_arr, att_maps_q.get("comb_img"), alpha=0.45)
                context['attn_final_img_b64'] = heatmap_to_base64_overlay(orig_arr, att_maps_q.get("final_patch_map"), alpha=0.45)

                # --- Attention highlights (text) ---
                tokens_decoded = tokenizer.convert_ids_to_tokens(txt_ids[0])
                context['attn_img2txt_html']  = attention_to_html(tokens_decoded, att_maps_q.get("img2txt")) if att_maps_q.get("img2txt") is not None else None
                context['attn_comb_txt_html'] = attention_to_html(tokens_decoded, att_maps_q.get("comb_txt")) if att_maps_q.get("comb_txt") is not None else None
                context['attn_final_txt_html']= attention_to_html(tokens_decoded, att_maps_q.get("final_token_map")) if att_maps_q.get("final_token_map") is not None else None
            except Exception as e:
                print(f"[WARN] query attention overlay failed: {e}")

        if main_target is not None and main_target in ig_maps_q:
            try:
                ig_q = ig_maps_q[main_target]
                context['ig_map_q_b64'] = heatmap_to_base64_overlay(orig_arr, ig_q, alpha=0.45)
            except Exception as e:
                print(f"[WARN] query IG overlay failed: {e}")

        if main_target is not None and main_target in gradcam_maps_q:
            try:
                gc_q = gradcam_maps_q[main_target]
                context['gradcam_map_q_b64'] = heatmap_to_base64_overlay(orig_arr, gc_q, alpha=0.45)
            except Exception as e:
                print(f"[WARN] query Grad-CAM overlay failed: {e}")

        # compute quick numeric comparisons for the query's own att vs ig
        image_attn_keys = ["txt2img", "comb_img", "final_patch_map"]
        context['attn_ig_metrics'] = {}
        if main_target is not None and main_target in ig_maps_q:
            for key in image_attn_keys:
                if att_maps_q.get(key) is not None:
                    try:
                        cm5  = compare_maps(att_maps_q[key], ig_maps_q[main_target], topk_frac=0.05)
                        cm20 = compare_maps(att_maps_q[key], ig_maps_q[main_target], topk_frac=0.20)
                        context['attn_ig_metrics'][key] = {
                            'pearson': round(cm5.get('pearson', 0.0), 4),
                            'spearman': round(cm5.get('spearman', 0.0), 4),
                            'iou_top5pct': round(cm5.get('iou_top5pct', 0.0), 4),
                            'iou_top20pct': round(cm20.get('iou_top20pct', 0.0), 4)
                        }
                    except Exception as e:
                        print(f"[WARN] compare_maps(query-{key}) failed: {e}")

        context['gc_attn_metrics'] = {}
        if main_target is not None and main_target in gradcam_maps_q:
            for key in image_attn_keys:
                if att_maps_q.get(key) is not None:
                    try:
                        cm5  = compare_maps(att_maps_q[key], gradcam_maps_q[main_target], topk_frac=0.05)
                        cm20 = compare_maps(att_maps_q[key], gradcam_maps_q[main_target], topk_frac=0.20)
                        context['gc_attn_metrics'][key] = {
                            'pearson': round(cm5.get('pearson', 0.0), 4),
                            'spearman': round(cm5.get('spearman', 0.0), 4),
                            'iou_top5pct': round(cm5.get('iou_top5pct', 0.0), 4),
                            'iou_top20pct': round(cm20.get('iou_top20pct', 0.0), 4)
                        }
                    except Exception as e:
                        print(f"[WARN] compare_maps(query-{key}) failed: {e}")

        # --- Preds / topk ---
        preds_arr = to_numpy(out.get("preds", None))
        context["preds"] = preds_arr[0] if preds_arr is not None and len(preds_arr) > 0 else None
        context["topk_idx"]  = topk_idx
        context["topk_vals"] = topk_vals
        context["topk_labels"] = [label_cols[i] for i in topk_idx] if topk_idx else []
        context["threshold"] = threshold
        context["pred_labels"] = [label_cols[i] for i, v in enumerate(context["preds"]) if v == 1] if context["preds"] is not None else []

        # --- Retrieval list (returned for the query) ---
        retrieval_pairs = list(zip(out.get("retrieval_ids", []), out.get("retrieval_dists", [])))
        context["retrieval"] = retrieval_pairs

        # Option flags
        show_image = "show_image" in request.form
        show_detail = "show_retrieval_detail" in request.form
        context["show_image"] = show_image
        context["show_retrieval_detail"] = show_detail

        # --- If user wants detailed retrieval: compute visuals & comparisons for each retrieved image ---
        retrieval_detailed = []
        retrieved_att_maps = []
        if show_detail and retrieval_pairs:
            for rid, dist in retrieval_pairs:
                try:
                    # labels & report for retrieved item
                    label_vec = labels_df.loc[rid][label_cols].astype(int).values
                    label_names = [label_cols[i] for i, v in enumerate(label_vec) if v == 1]
                    report_text = report_lookup.get(rid, "No report found")

                    # find dicom and preprocess
                    dcm_path = find_dicom_file(rid)
                    dcm_tensor = preproc(dcm_path)  # (C, H, W)
                    orig_arr_r = dcm_tensor.squeeze().cpu().numpy()
                    if orig_arr_r.dtype != np.float32 and orig_arr_r.max() > 1.0:
                        orig_arr_r = orig_arr_r.astype(np.float32) / 255.0

                    # compute image bytes for UI
                    img_b64 = np_to_base64_img(dcm_tensor.numpy(), cmap="gray")

                    att_maps_r, ig_maps_r, gradcam_maps_r = {}, {}, {}

                    try:
                        out_ret = model.get_explain_score(
                            dcm_tensor.unsqueeze(0).to(device),
                            txt_ids, txt_mask, main_target
                        )

                        # Use same keys/structure as query
                        att_maps_r = out_ret.get("attention_map", {})
                        ig_maps_r      = out_ret.get("ig_maps", {})
                        gradcam_maps_r = out_ret.get("gradcam_maps", {})

                    except Exception as e:
                        print(f"[WARN] failed to compute explanations for retrieved {rid}: {e}")


                    # create base64 overlays for retrieved item
                    attn_txt2img_b64_r, attn_comb_img_b64_r, attn_final_img_b64_r = None, None, None
                    attn_img2txt_html_r, attn_comb_txt_html_r, attn_final_txt_html_r = None, None, None
                    ig_b64_r, gradcam_b64_r = None, None
                    ig_map_r, gradcam_map_r = None, None

                    try:
                        if att_maps_r.get("txt2img") is not None:
                            attn_txt2img_b64_r = heatmap_to_base64_overlay(orig_arr_r, att_maps_r["txt2img"], alpha=0.45)

                        if att_maps_r.get("comb_img") is not None:
                            attn_comb_img_b64_r = heatmap_to_base64_overlay(orig_arr_r, att_maps_r["comb_img"], alpha=0.45)

                        if att_maps_r.get("final_patch_map") is not None:
                            attn_final_img_b64_r = heatmap_to_base64_overlay(orig_arr_r, att_maps_r["final_patch_map"], alpha=0.45)

                        # Text highlights (HTML rendering)
                        tokens_r = tokenizer(
                            report_text,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=cfg.text_dim
                        )
                        tokens_decoded_r = tokenizer.convert_ids_to_tokens(tokens_r.input_ids[0])

                        if att_maps_r.get("img2txt") is not None:
                            attn_img2txt_html_r = attention_to_html(tokens_decoded_r, att_maps_r["img2txt"])

                        if att_maps_r.get("comb_txt") is not None:
                            attn_comb_txt_html_r = attention_to_html(tokens_decoded_r, att_maps_r["comb_txt"])

                        if att_maps_r.get("final_token_map") is not None:
                            attn_final_txt_html_r = attention_to_html(tokens_decoded_r, att_maps_r["final_token_map"])

                    except Exception as e:
                        print(f"[WARN] render retrieved attention failed for {rid}: {e}")

                    if att_maps_r.get("final_patch_map") is not None:
                        retrieved_att_maps.append(att_maps_r["final_patch_map"])

                    if main_target is not None and ig_maps_r.get(main_target) is not None:
                        try:
                            ig_map_r = ig_maps_r[main_target]
                            ig_b64_r = heatmap_to_base64_overlay(orig_arr_r, ig_map_r, alpha=0.45)
                        except Exception as e:
                            print(f"[WARN] render retrieved IG overlay {rid} failed: {e}")

                    if main_target is not None and gradcam_maps_r.get(main_target) is not None:
                        try:
                            gradcam_map_r = gradcam_maps_r[main_target]
                            gradcam_b64_r = heatmap_to_base64_overlay(orig_arr_r, gradcam_map_r, alpha=0.45)
                        except Exception as e:
                            print(f"[WARN] render retrieved Grad-CAM overlay {rid} failed: {e}")

                    # compute cross-image comparisons (query vs retrieved)-
                    cross_metrics = {}
                    try:
                        # --- Attention maps ---
                        for att_type in ["txt2img", "comb_img", "final_patch_map"]:
                            if att_maps_q.get(att_type) is not None and att_maps_r.get(att_type) is not None:
                                cm_5  = compare_maps(att_maps_q[att_type], att_maps_r[att_type], topk_frac=0.05)
                                cm_20 = compare_maps(att_maps_q[att_type], att_maps_r[att_type], topk_frac=0.20)

                                cross_metrics[f"att_{att_type}_pearson_top5pct"]  = round(cm_5.get("pearson", 0.0), 4)
                                cross_metrics[f"att_{att_type}_spearman_top5pct"] = round(cm_5.get("spearman", 0.0), 4)
                                cross_metrics[f"att_{att_type}_iou_top5pct"]      = round(cm_5.get("iou_top5pct", 0.0), 4)
                                cross_metrics[f"att_{att_type}_iou_top20pct"]     = round(cm_20.get("iou_top20pct", 0.0), 4)

                        # --- Integrated Gradients (per target) ---
                        if main_target is not None and main_target in ig_maps_q and ig_map_r is not None:
                            cm_ig_5  = compare_maps(ig_maps_q[main_target], ig_map_r, topk_frac=0.05)
                            cm_ig_20 = compare_maps(ig_maps_q[main_target], ig_map_r, topk_frac=0.20)
                            cross_metrics["ig_pearson_top5pct"]  = round(cm_ig_5.get("pearson", 0.0), 4)
                            cross_metrics["ig_spearman_top5pct"] = round(cm_ig_5.get("spearman", 0.0), 4)
                            cross_metrics["ig_iou_top5pct"]      = round(cm_ig_5.get("iou_top5pct", 0.0), 4)
                            cross_metrics["ig_iou_top20pct"]     = round(cm_ig_20.get("iou_top20pct", 0.0), 4)

                        # --- Grad-CAM (per target) ---
                        if main_target is not None and main_target in gradcam_maps_q and gradcam_map_r is not None:
                            cm_gc_5  = compare_maps(gradcam_maps_q[main_target], gradcam_map_r, topk_frac=0.05)
                            cm_gc_20 = compare_maps(gradcam_maps_q[main_target], gradcam_map_r, topk_frac=0.20)
                            cross_metrics["gradcam_pearson_top5pct"]  = round(cm_gc_5.get("pearson", 0.0), 4)
                            cross_metrics["gradcam_spearman_top5pct"] = round(cm_gc_5.get("spearman", 0.0), 4)
                            cross_metrics["gradcam_iou_top5pct"]      = round(cm_gc_5.get("iou_top5pct", 0.0), 4)
                            cross_metrics["gradcam_iou_top20pct"]     = round(cm_gc_20.get("iou_top20pct", 0.0), 4)

                    except Exception as e:
                        print(f"[WARN] cross-compare failed for {rid}: {e}")

                    except Exception as e:
                        print(f"[WARN] cross-compare failed for {rid}: {e}")

                    retrieval_detailed.append({
                        "id": rid,
                        "dist": dist,
                        "labels": label_names,
                        "report": report_text,
                        "image": img_b64,
                        "attn_txt2img_b64_r": attn_txt2img_b64_r,
                        "attn_comb_img_b64_r": attn_comb_img_b64_r,
                        "attn_final_img_b64_r": attn_final_img_b64_r,
                        "attn_img2txt_html_r": attn_img2txt_html_r,
                        "attn_comb_txt_html_r": attn_comb_txt_html_r,
                        "attn_final_txt_html_r": attn_final_txt_html_r,
                        "ig_b64": ig_b64_r,
                        "gradcam_b64": gradcam_b64_r,
                        "cross_metrics": cross_metrics
                    })

                except Exception as e:
                    print(f"[WARN] Retrieval detail failed for {rid}: {e}")

            if len(retrieved_att_maps) > 1:
                overlaps = []
                for i in range(len(retrieved_att_maps)):
                    for j in range(i + 1, len(retrieved_att_maps)):
                        cm = compare_maps(retrieved_att_maps[i], retrieved_att_maps[j], topk_frac=0.05)
                        overlaps.append(cm['iou_top5pct'])
                avg_overlap = np.mean(overlaps)
                avg_diversity = 1 - avg_overlap
                context['retrieval_diversity_score'] = round(avg_diversity, 4)
            else:
                context['retrieval_diversity_score'] = None

            # --- Compute overlap/diversity restricted to same-class retrieved items ---
            same_class_maps = []
            for item, amap in zip(retrieval_detailed, retrieved_att_maps):
                if main_target is not None and label_cols[main_target] in item["labels"]:
                    same_class_maps.append(amap)

            if len(same_class_maps) > 1:
                overlaps = []
                for i in range(len(same_class_maps)):
                    for j in range(i + 1, len(same_class_maps)):
                        cm = compare_maps(same_class_maps[i], same_class_maps[j], topk_frac=0.05)
                        overlaps.append(cm['iou_top5pct'])
                avg_overlap = np.mean(overlaps)
                avg_diversity = 1 - avg_overlap
                context['retrieval_same_class_overlap'] = round(avg_overlap, 4)
                context['retrieval_same_class_diversity'] = round(avg_diversity, 4)
            else:
                context['retrieval_same_class_overlap'] = None
                context['retrieval_same_class_diversity'] = None
                
        context["retrieval_detailed"] = retrieval_detailed

    # Debug prints
    print("Context keys:", list(context.keys()))
    print("Retrieval pairs:", context.get("retrieval"))

    return render_template("index.html", **context)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
