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

from config import Config
from tensorDICOM import DICOMImagePreprocessor
from model import MultiModalRetrievalModel
from labeledData import disease_groups, normal_groups
from dataParser import parse_openi_xml
from retrieval import make_retrieval_engine
from helper import compare_maps, heatmap_to_base64_overlay
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

        # Normalize outputs
        att_map_q = to_numpy(out.get('attention_map', None))    # (H,W) np or None
        ig_maps_q  = out.get('ig_maps', {}) or {}               # dict target->np
        ig_maps_q = {int(k): to_numpy(v) for k,v in ig_maps_q.items()}

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

        context['attn_map_q_b64'] = None
        context['ig_map_q_b64']   = None
        context['hybrid_q_b64']   = None

        if att_map_q is not None:
            try:
                context['attn_map_q_b64'] = heatmap_to_base64_overlay(orig_arr, att_map_q, alpha=0.45)
            except Exception as e:
                print(f"[WARN] query attention overlay failed: {e}")

        if main_target is not None and main_target in ig_maps_q:
            try:
                ig_q = ig_maps_q[main_target]
                context['ig_map_q_b64'] = heatmap_to_base64_overlay(orig_arr, ig_q, alpha=0.45)
            except Exception as e:
                print(f"[WARN] query IG overlay failed: {e}")

        if att_map_q is not None and main_target is not None and main_target in ig_maps_q:
            try:
                hybrid_q = (att_map_q * ig_maps_q[main_target])
                hybrid_q = (hybrid_q - hybrid_q.min()) / (hybrid_q.max() - hybrid_q.min() + 1e-8)
                context['hybrid_q_b64'] = heatmap_to_base64_overlay(orig_arr, hybrid_q, alpha=0.55)
            except Exception as e:
                print(f"[WARN] query hybrid overlay failed: {e}")

        # compute quick numeric comparisons for the query's own att vs ig
        if att_map_q is not None and main_target is not None and main_target in ig_maps_q:
            try:
                metrics_5 = compare_maps(att_map_q, ig_maps_q[main_target], topk_frac=0.05)
                metrics_20 = compare_maps(att_map_q, ig_maps_q[main_target], topk_frac=0.20)
                context['explain_metrics'] = {
                    'pearson': round(metrics_5.get('pearson', 0.0), 4),
                    'spearman': round(metrics_5.get('spearman', 0.0), 4),
                    'iou_top5pct': round(metrics_5.get('iou_top5pct', 0.0), 4),
                    'iou_top20pct': round(metrics_20.get('iou_top20pct', 0.0), 4),
                    'predict_time_s': round(predict_time, 3)
                }
            except Exception as e:
                print(f"[WARN] compare_maps(query) failed: {e}")
                context['explain_metrics'] = None
        else:
            context['explain_metrics'] = None

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
        if show_detail and retrieval_pairs:
            for rid, dist in retrieval_pairs:
                try:
                    # labels & report for retrieved item
                    if rid not in labels_df.index:
                        raise KeyError(f"{rid} not in label CSV")
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

                    # compute explanation maps for retrieved image using same text query
                    with torch.no_grad():
                        # forward to collect attention weights on retrieved image
                        _, _, attn_weights_ret = model.forward(
                            dcm_tensor.unsqueeze(0).to(device),
                            txt_ids, txt_mask,
                            return_attention=True
                        )

                    # extract last layer txt2img attention tensor if available
                    att_map_r = None
                    try:
                        layer_key = f"layer_{len(model.fusion_layers) - 1}_txt2img"
                        att_tensor = attn_weights_ret.get(layer_key, None)
                        if att_tensor is not None:
                            # pass through explanation engine helper to produce HxW map
                            G = int(att_tensor.shape[-1] ** 0.5)
                            att_map_r = model.explainer.compute_attention_map(att_tensor, grid_size=G)
                    except Exception as e:
                        print(f"[WARN] failed to compute retrieved attention for {rid}: {e}")
                        att_map_r = None

                    # compute IG map for the same main_target (if available)
                    ig_map_r = None
                    if main_target is not None:
                        try:
                            # extract features for retrieved image
                            (img_global_r, img_patches_r), txt_feats_r = model.backbones(
                                dcm_tensor.unsqueeze(0).to(device),
                                txt_ids, txt_mask
                            )
                            ig_map_r = model.explainer.compute_ig_map_for_target(
                                img_global_r[0:1], img_patches_r[0:1], txt_feats_r[0:1], int(main_target)
                            )
                        except Exception as e:
                            print(f"[WARN] IG for retrieved {rid} failed: {e}")
                            ig_map_r = None

                    # create base64 overlays
                    attn_b64_r = None
                    ig_b64_r = None
                    hybrid_b64_r = None
                    if att_map_r is not None:
                        try:
                            attn_b64_r = heatmap_to_base64_overlay(orig_arr_r, att_map_r, alpha=0.45)
                        except Exception as e:
                            print(f"[WARN] render retrieved attn overlay {rid} failed: {e}")
                    if ig_map_r is not None:
                        try:
                            ig_b64_r = heatmap_to_base64_overlay(orig_arr_r, ig_map_r, alpha=0.45)
                        except Exception as e:
                            print(f"[WARN] render retrieved IG overlay {rid} failed: {e}")
                    if att_map_r is not None and ig_map_r is not None:
                        try:
                            hybrid_r = (att_map_r * ig_map_r)
                            hybrid_r = (hybrid_r - hybrid_r.min()) / (hybrid_r.max() - hybrid_r.min() + 1e-8)
                            hybrid_b64_r = heatmap_to_base64_overlay(orig_arr_r, hybrid_r, alpha=0.55)
                        except Exception as e:
                            print(f"[WARN] render retrieved hybrid overlay {rid} failed: {e}")

                    # compute cross-image comparisons (query vs retrieved)
                    cross_metrics = {}
                    try:
                        if att_map_q is not None and att_map_r is not None:
                            # compare attention maps
                            cm_att_5 = compare_maps(att_map_q, att_map_r, topk_frac=0.05)
                            cm_att_20 = compare_maps(att_map_q, att_map_r, topk_frac=0.20)
                            cross_metrics['att_pearson_top5pct'] = round(cm_att_5.get('pearson', 0.0), 4)
                            cross_metrics['att_spearman_top5pct'] = round(cm_att_5.get('spearman', 0.0), 4)
                            cross_metrics['att_iou_top5pct'] = round(cm_att_5.get('iou_top5pct', 0.0), 4)
                            cross_metrics['att_iou_top20pct'] = round(cm_att_20.get('iou_top20pct', 0.0), 4)
                        if main_target is not None and (main_target in ig_maps_q) and ig_map_r is not None:
                            cm_ig_5 = compare_maps(ig_maps_q[main_target], ig_map_r, topk_frac=0.05)
                            cm_ig_20 = compare_maps(ig_maps_q[main_target], ig_map_r, topk_frac=0.20)
                            cross_metrics['ig_pearson_top5pct'] = round(cm_ig_5.get('pearson', 0.0), 4)
                            cross_metrics['ig_spearman_top5pct'] = round(cm_ig_5.get('spearman', 0.0), 4)
                            cross_metrics['ig_iou_top5pct'] = round(cm_ig_5.get('iou_top5pct', 0.0), 4)
                            cross_metrics['ig_iou_top20pct'] = round(cm_ig_20.get('iou_top20pct', 0.0), 4)
                    except Exception as e:
                        print(f"[WARN] cross-compare failed for {rid}: {e}")

                    retrieval_detailed.append({
                        "id": rid,
                        "dist": dist,
                        "labels": label_names,
                        "report": report_text,
                        "image": img_b64,
                        "attn_b64": attn_b64_r,
                        "ig_b64": ig_b64_r,
                        "hybrid_b64": hybrid_b64_r,
                        "cross_metrics": cross_metrics
                    })

                except Exception as e:
                    print(f"[WARN] Retrieval detail failed for {rid}: {e}")

        context["retrieval_detailed"] = retrieval_detailed

    # Debug prints
    print("Context keys:", list(context.keys()))
    print("Retrieval pairs:", context.get("retrieval"))

    return render_template("index.html", **context)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
