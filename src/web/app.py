from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))

import io
import base64
import pickle
import time
from typing import Optional

from flask import Flask, render_template, request
import numpy as np
import torch
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
BASE_DIR       = Path(__file__).resolve().parent.parent.parent
CKPT_DIR       = BASE_DIR / "checkpoints"
MODEL_DIR      = BASE_DIR / "models"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
OUTPUT_DIR     = BASE_DIR / "outputs"
XML_DIR        = BASE_DIR / "data" / "openi" / "xml" / "NLMCXR_reports" / "ecgen-radiology"
DICOM_ROOT     = BASE_DIR / "data" / "openi" / "dicom"
CONFIG_DIR     = BASE_DIR / "configs"
CKPT_PATH      = CKPT_DIR / "model_best.pt"
REPORT_CACHE_PATH = OUTPUT_DIR / "openi_reports.pkl"
#────────────────────────────────────────────────────────────────────────────────

# load config
cfg = Config.load(CONFIG_DIR / "config.yaml")

app = Flask(__name__, static_folder="static", template_folder="templates")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# labels
label_cols = list(disease_groups.keys()) + list(normal_groups.keys()) + list(finding_groups.keys()) + list(symptom_groups.keys())
labels_df = pd.read_csv(OUTPUT_DIR / "openi_labels_final.csv").set_index("id")

# Lazy-initialized globals (heavy objects)
report_lookup: Optional[dict] = None
retriever = None
model = None
preproc = None
tokenizer = None

def load_report_lookup_via_parser(xml_dir, dicom_root) -> dict:
    if REPORT_CACHE_PATH.exists():
        print(f"[INFO] Loading cached report lookup from {REPORT_CACHE_PATH.name}")
        with open(REPORT_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    # combined_groups should be a list of label names
    combined_groups = list(disease_groups.keys()) + list(normal_groups.keys()) + list(finding_groups.keys()) + list(symptom_groups.keys())
    print("[INFO] Parsing reports using parse_openi_xml()...")
    parsed_records = parse_openi_xml(xml_dir, dicom_root, combined_groups=combined_groups)
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
    primary = list(Path(DICOM_ROOT).rglob(f"{rid}.dcm"))
    if primary:
        return primary[0]

    # Try fallback without leading patient ID
    parts = rid.split("_")
    fallback_id = "_".join(parts[1:]) if len(parts) > 1 else rid
    fallback = list(Path(DICOM_ROOT).rglob(f"{fallback_id}.dcm"))
    if fallback:
        print(f"[INFO] Using fallback DICOM path: {fallback[0].name}")
        return fallback[0]

    raise FileNotFoundError(f"DICOM not found for either {rid}.dcm or {fallback_id}.dcm")

def init_heavy_resources():
    """Initialize heavy resources only once (called lazily)."""
    global report_lookup, retriever, model, preproc, tokenizer

    if report_lookup is None:
        report_lookup = load_report_lookup_via_parser(XML_DIR, DICOM_ROOT)

    if retriever is None:
        retriever = make_retrieval_engine(
            features_path=str(EMBEDDINGS_DIR / "train_joint_embeddings.npy"),
            ids_path=str(EMBEDDINGS_DIR / "train_ids.json"),
            method="dls",
            link_threshold=0.5,
            max_links=10
        )

    if model is None:
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

    if preproc is None:
        preproc = DICOMImagePreprocessor()

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        except Exception as e:
            print(f"[WARN] Failed to load ClinicalBERT tokenizer: {e}. Falling back to bert-base-uncased tokenizer.")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("[INFO] Heavy resources initialized.")

def normalize_for_display(img_np: np.ndarray) -> np.ndarray:
    """Return HxW or HxWxC array scaled into [0,1] for display."""
    if hasattr(img_np, "detach"):
        img_np = img_np.detach().cpu().numpy()
    img = img_np.copy()
    # channel-first to HWC
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = img[:, :, 0]
    if np.issubdtype(img.dtype, np.integer) or img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    minv, maxv = float(np.nanmin(img)), float(np.nanmax(img))
    if maxv - minv > 1e-6:
        img = (img - minv) / (maxv - minv)
    else:
        img = np.clip(img, 0.0, 1.0)
    return img

def np_to_base64_img(arr: np.ndarray, cmap="gray") -> str:
    fig, ax = plt.subplots(figsize=(4, 4))

    # If a torch tensor was passed in, ensure it's numpy
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()

    # Convert channel-first (C,H,W) to H,W,C for plotting
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]

    if arr.ndim == 2:
        ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)
    else:
        ax.imshow(arr, vmin=0.0, vmax=1.0)

    ax.axis("off")
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def resize_to_match(src_map, target_shape):
    """Resize src_map (H,W) to target_shape (H2,W2) using bilinear interpolation."""
    # src_map: 2D numpy array
    src_tensor = torch.tensor(src_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    # target_shape must be (H2, W2)
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

def _overlay_safe(orig_arr, amap, alpha=0.45):
    """Resize amap if needed and call heatmap_to_base64_overlay safely.

    orig_arr: HxW or HxWxC numpy array (display image)
    amap: 2D heatmap (H2,W2) or tensor
    """
    if amap is None:
        return None
    amap_np = to_numpy(amap)
    if amap_np is None:
        return None

    # Ensure orig_arr is HxW for target_shape computation
    if orig_arr.ndim == 3:
        # orig could be HxWxC -> spatial dims = orig_arr.shape[:2]
        target_shape = orig_arr.shape[:2]
    elif orig_arr.ndim == 2:
        target_shape = orig_arr.shape
    else:
        # fallback: try last two dims
        target_shape = orig_arr.shape[-2:]

    # If amap is 2D and shapes differ, resize to target_shape (H,W)
    if amap_np.ndim == 2 and amap_np.shape != tuple(target_shape):
        try:
            amap_np = resize_to_match(amap_np, target_shape)
        except Exception as e:
            print(f"[WARN] resize_to_match failed: {e}")
            return None

    try:
        return heatmap_to_base64_overlay(orig_arr, amap_np, alpha=alpha)
    except Exception as e:
        print(f"[WARN] heatmap overlay failed: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    # lazy init heavy resources on first request (avoids double work with flask reloader)
    if report_lookup is None or retriever is None or model is None or preproc is None or tokenizer is None:
        init_heavy_resources()

    context = {}
    if request.method == "POST":
        print("[INFO] POST request")
        # --- Load inputs ---
        if "dicom_file" not in request.files:
            return render_template("index.html", error="Please upload a DICOM file.")

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
            max_length=getattr(cfg, "text_dim", 128)
        )
        txt_ids  = tokens.input_ids.to(device)
        txt_mask = tokens.attention_mask.to(device)
        
        # prepare orig_arr for visualization (H,W) scaled 0..1
        orig_arr = raw_tensor.squeeze().cpu().numpy()   # could be (C,H,W) or (H,W) or (H,W,C)
        # If channel-first C,H,W convert to H,W or H,W,C
        if orig_arr.ndim == 3 and orig_arr.shape[0] in (1, 3):
            # C,H,W -> H,W,C then if single channel squeeze to H,W
            orig_arr = np.transpose(orig_arr, (1, 2, 0))
            if orig_arr.shape[2] == 1:
                orig_arr = orig_arr[:, :, 0]

        # If integer 0..255 -> scale to [0,1]
        if np.issubdtype(orig_arr.dtype, np.integer) or orig_arr.max() > 1.0:
            orig_arr = orig_arr.astype(np.float32) / 255.0

        # If preprocessing produced values outside [0,1] (e.g. mean-subtracted), rescale to [0,1]
        minv, maxv = float(np.nanmin(orig_arr)), float(np.nanmax(orig_arr))
        if maxv - minv > 1e-6:
            orig_arr_disp = (orig_arr - minv) / (maxv - minv)
        else:
            orig_arr_disp = np.clip(orig_arr, 0.0, 1.0)

        # Use orig_arr_disp for plotting/overlaying (keeps original orig_arr if needed elsewhere)
        orig_arr = orig_arr_disp

        # --- Predict + Explain for query image ---
        start_time = time.perf_counter()
        out = model.predict(img_tensor, txt_ids, txt_mask, K=5, explain=True)
        predict_time = time.perf_counter() - start_time

        # Get explaining outputs
        att_maps_q = out.get("attention_map", {}) or {}
        att_maps_q = {k: to_numpy(v) for k, v in att_maps_q.items() if v is not None}
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
        context['img_orig'] = np_to_base64_img(orig_arr, cmap="gray")
        context["text_query"] = text_query
        context['attn_map_q_b64'] = None
        context['ig_map_q_b64']   = None
        context['gradcam_map_q_b64'] = None

        if att_maps_q is not None:
            try:
                # --- Attention overlays ---
                context['attn_txt2img_b64']   = _overlay_safe(orig_arr, att_maps_q.get("txt2img"), alpha=0.45)
                context['attn_comb_img_b64']  = _overlay_safe(orig_arr, att_maps_q.get("comb_img"), alpha=0.45)
                context['attn_final_img_b64'] = _overlay_safe(orig_arr, att_maps_q.get("final_patch_map"), alpha=0.45)

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
                context['ig_map_q_b64'] = _overlay_safe(orig_arr, ig_q, alpha=0.45)
            except Exception as e:
                print(f"[WARN] query IG overlay failed: {e}")

        if main_target is not None and main_target in gradcam_maps_q:
            try:
                gc_q = gradcam_maps_q[main_target]
                context['gradcam_map_q_b64'] = _overlay_safe(orig_arr, gc_q, alpha=0.45)
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
        raw_rids = out.get("retrieval_ids", []) or []
        raw_dists = out.get("retrieval_dists", []) or []

        # If the retriever returned batch-style lists (e.g. [[id1,id2,...]]), flatten for batch=1
        if len(raw_rids) == 1 and isinstance(raw_rids[0], (list, tuple)):
            ids = list(raw_rids[0])
            dists = list(raw_dists[0]) if raw_dists and isinstance(raw_dists[0], (list, tuple)) else list(raw_dists)
        else:
            ids = list(raw_rids)
            dists = list(raw_dists)

        retrieval_pairs = list(zip(ids, dists))
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
                    if rid not in labels_df.index:
                        print(f"[WARN] Retrieved id not found in labels_df: {rid}")
                        continue

                    # labels & report for retrieved item
                    label_vec = labels_df.loc[rid][label_cols].astype(int).values
                    label_names = [label_cols[i] for i, v in enumerate(label_vec) if v == 1]
                    report_text = report_lookup.get(rid, "No report found")

                    # find dicom and preprocess
                    dcm_path = find_dicom_file(rid)
                    dcm_tensor = preproc(dcm_path)  # (C, H, W)
                    orig_arr_r = dcm_tensor.squeeze().cpu().numpy()
                    orig_arr_r = normalize_for_display(orig_arr_r)

                    # compute image bytes for UI (use normalized display array)
                    img_b64 = np_to_base64_img(orig_arr_r, cmap="gray")

                    # === Get explanations for retrieved item ===
                    print(f"[DEBUG] Computing explanations for retrieved id={rid}, dist={dist}")
                    try:
                        out_ret = model.predict(
                            dcm_tensor.unsqueeze(0).to(device),
                            txt_ids, txt_mask,
                            K=5,
                            explain=True
                        )
                        print("[DEBUG] out_ret keys:", list(out_ret.keys()))
                    except Exception as e:
                        print(f"[WARN] failed to compute explanations for retrieved {rid}: {e}")
                        out_ret = {}

                    # Normalize and convert returned maps to numpy
                    att_maps_r = out_ret.get("attention_map", {}) or {}
                    att_maps_r = {k: to_numpy(v) for k, v in att_maps_r.items() if v is not None}

                    ig_maps_r = out_ret.get("ig_maps", {}) or {}
                    ig_maps_r = {int(k): to_numpy(v) for k, v in ig_maps_r.items()}

                    gradcam_maps_r = out_ret.get("gradcam_maps", {}) or {}
                    gradcam_maps_r = {int(k): to_numpy(v) for k, v in gradcam_maps_r.items()}

                    # debug shapes for server log
                    for k, v in att_maps_r.items():
                        try:
                            print(f"[DEBUG] att_maps_r[{k}] shape: {np.shape(v)}")
                        except Exception:
                            print(f"[DEBUG] att_maps_r[{k}] type: {type(v)}")
                    for k, v in ig_maps_r.items():
                        print(f"[DEBUG] ig_maps_r[{k}] shape: {np.shape(v)}")
                    for k, v in gradcam_maps_r.items():
                        print(f"[DEBUG] gradcam_maps_r[{k}] shape: {np.shape(v)}")

                    # create base64 overlays using names expected by the template
                    attn_txt_b64 = _overlay_safe(orig_arr_r, att_maps_r.get("txt2img"), alpha=0.45)
                    attn_comb_b64 = _overlay_safe(orig_arr_r, att_maps_r.get("comb_img"), alpha=0.45)
                    attn_final_b64 = _overlay_safe(orig_arr_r, att_maps_r.get("final_patch_map"), alpha=0.45)
                    retrieved_att_maps.append(att_maps_r.get("final_patch_map"))

                    # Text-level HTML attentions (template expects attn_img_html, comb_txt_html, final_token_html)
                    attn_img_html = None
                    comb_txt_html = None
                    final_token_html = None
                    try:
                        tokens_r = tokenizer(
                            report_text,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=getattr(cfg, "text_dim", 128)
                        )
                        tokens_decoded_r = tokenizer.convert_ids_to_tokens(tokens_r.input_ids[0])

                        if att_maps_r.get("img2txt") is not None:
                            attn_img_html = attention_to_html(tokens_decoded_r, att_maps_r["img2txt"])

                        if att_maps_r.get("comb_txt") is not None:
                            comb_txt_html = attention_to_html(tokens_decoded_r, att_maps_r["comb_txt"])

                        if att_maps_r.get("final_token_map") is not None:
                            final_token_html = attention_to_html(tokens_decoded_r, att_maps_r["final_token_map"])
                    except Exception as e:
                        print(f"[WARN] render retrieved attention (text) failed for {rid}: {e}")

                    # IG / Grad-CAM overlays for the main_target (if available)
                    ig_b64_r = None
                    gradcam_b64_r = None
                    if main_target is not None:
                        if ig_maps_r.get(main_target) is not None:
                            ig_b64_r = _overlay_safe(orig_arr_r, ig_maps_r[main_target], alpha=0.45)
                        if gradcam_maps_r.get(main_target) is not None:
                            gradcam_b64_r = _overlay_safe(orig_arr_r, gradcam_maps_r[main_target], alpha=0.45)

                    # cross-image metrics (keep your existing logic here, ensure cross_metrics exists)
                    cross_metrics = {}
                    try:
                        for att_type in ["txt2img", "comb_img", "final_patch_map"]:
                            if att_maps_q.get(att_type) is not None and att_maps_r.get(att_type) is not None:
                                cm_5  = compare_maps(att_maps_q[att_type], att_maps_r[att_type], topk_frac=0.05)
                                cm_20 = compare_maps(att_maps_q[att_type], att_maps_r[att_type], topk_frac=0.20)

                                cross_metrics[f"att_{att_type}_pearson_top5pct"]  = round(cm_5.get("pearson", 0.0), 4)
                                cross_metrics[f"att_{att_type}_spearman_top5pct"] = round(cm_5.get("spearman", 0.0), 4)
                                cross_metrics[f"att_{att_type}_iou_top5pct"]      = round(cm_5.get("iou_top5pct", 0.0), 4)
                                cross_metrics[f"att_{att_type}_iou_top20pct"]     = round(cm_20.get("iou_top20pct", 0.0), 4)

                        if main_target is not None and main_target in ig_maps_q and ig_b64_r is not None:
                            cm_ig_5  = compare_maps(ig_maps_q[main_target], ig_maps_r[main_target], topk_frac=0.05)
                            cm_ig_20 = compare_maps(ig_maps_q[main_target], ig_maps_r[main_target], topk_frac=0.20)
                            cross_metrics["ig_pearson_top5pct"]  = round(cm_ig_5.get("pearson", 0.0), 4)
                            cross_metrics["ig_spearman_top5pct"] = round(cm_ig_5.get("spearman", 0.0), 4)
                            cross_metrics["ig_iou_top5pct"]      = round(cm_ig_5.get("iou_top5pct", 0.0), 4)
                            cross_metrics["ig_iou_top20pct"]     = round(cm_ig_20.get("iou_top20pct", 0.0), 4)

                        if main_target is not None and main_target in gradcam_maps_q and gradcam_b64_r is not None:
                            cm_gc_5  = compare_maps(gradcam_maps_q[main_target], gradcam_maps_r[main_target], topk_frac=0.05)
                            cm_gc_20 = compare_maps(gradcam_maps_q[main_target], gradcam_maps_r[main_target], topk_frac=0.20)
                            cross_metrics["gradcam_pearson_top5pct"]  = round(cm_gc_5.get("pearson", 0.0), 4)
                            cross_metrics["gradcam_spearman_top5pct"] = round(cm_gc_5.get("spearman", 0.0), 4)
                            cross_metrics["gradcam_iou_top5pct"]      = round(cm_gc_5.get("iou_top5pct", 0.0), 4)
                            cross_metrics["gradcam_iou_top20pct"]     = round(cm_gc_20.get("iou_top20pct", 0.0), 4)
                    except Exception as e:
                        print(f"[WARN] cross-compare failed for {rid}: {e}")

                    # Append with keys that match your Jinja template
                    retrieval_detailed.append({
                        "id": rid,
                        "dist": dist,
                        "labels": label_names,
                        "report": report_text,
                        "image": img_b64,
                        "attn_txt_b64": attn_txt_b64,
                        "attn_comb_b64": attn_comb_b64,
                        "attn_final_b64": attn_final_b64,
                        "attn_img_html": attn_img_html,
                        "comb_txt_html": comb_txt_html,
                        "final_token_html": final_token_html,
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
    # To avoid double initialization during development on Windows, disable flask reloader.
    # You can set debug=True but use_reloader=False so the process only initializes once.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
