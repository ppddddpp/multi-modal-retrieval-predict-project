import io
import base64
import sys
import pickle
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from flask import Flask, render_template, request
import numpy as np
import torch
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
    """
    Search for the DICOM file by ID or fallback pattern recursively under DICOM_ROOT.
    """
    primary = list(DICOM_ROOT.rglob(f"{rid}.dcm"))
    if primary:
        return primary[0]

    # Try fallback without leading patient ID (e.g., IM-0633-1001)
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

@app.route("/", methods=["GET", "POST"])
def index():
    context = {}
    if request.method == "POST":
        # DICOM + text
        dcm_bytes  = request.files["dicom_file"].read()
        img_tensor = preproc(dcm_bytes).unsqueeze(0).to(device)
        threshold = float(request.form.get("threshold", 0.5))
        tokens    = tokenizer(
            request.form["text_query"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        txt_ids  = tokens.input_ids.to(device)
        txt_mask = tokens.attention_mask.to(device)

        # Predict + explain
        out = model.predict(img_tensor, txt_ids, txt_mask, threshold=threshold, K=5, explain=True)

        pred_labels = [label_cols[i] for i, v in enumerate(out["preds"][0]) if v == 1]
        context["pred_labels"] = pred_labels


        # Show image option and retrieval detail
        show_image = "show_image" in request.form
        show_detail = "show_retrieval_detail" in request.form
        context["show_image"] = show_image
        context["show_retrieval_detail"] = show_detail

        # Assemble context
        context["preds"] = out["preds"][0]
        context["topk_idx"]  = out["topk_idx"][0]
        context["topk_vals"] = out["topk_vals"][0]
        context["topk_labels"] = [label_cols[i] for i in context["topk_idx"]]
        context["threshold"] = threshold

        topk_labeled_probabilities = []
        for i in range(len(context["topk_idx"])):
            label = context["topk_labels"][i]
            prob = context["topk_vals"][i]
            topk_labeled_probabilities.append({'label': label, 'prob': prob})
        context["topk_labels_with_probs"] = topk_labeled_probabilities

        # Retrieval
        context["retrieval"] = list(zip(out["retrieval_ids"], out["retrieval_dists"]))
        if context["show_retrieval_detail"]:
            detailed = []
            for rid, dist in zip(out["retrieval_ids"], out["retrieval_dists"]):
                try:
                    if rid not in labels_df.index:
                        raise KeyError(f"{rid} not in label CSV")
                    label_vec = labels_df.loc[rid][label_cols].astype(int).values
                    label_names = [label_cols[i] for i, v in enumerate(label_vec) if v == 1]

                    report_text = report_lookup.get(rid, "No report found")

                    dcm_path = find_dicom_file(rid)
                    if not dcm_path.exists():
                        fallback_id = "_".join(rid.split("_")[1:])
                        fallback_path = DICOM_ROOT / f"{fallback_id}.dcm"
                        if fallback_path.exists():
                            print(f"[INFO] Using fallback DICOM path: {fallback_path.name}")
                            dcm_path = fallback_path
                        else:
                            raise FileNotFoundError(f"DICOM not found for either {rid}.dcm or {fallback_id}.dcm")

                    dcm_tensor = preproc(dcm_path).numpy()
                    img = np_to_base64_img(dcm_tensor, cmap="gray")

                    detailed.append({
                        "id": rid,
                        "dist": dist,
                        "labels": label_names,
                        "report": report_text,
                        "image": img
                    })

                except Exception as e:
                    print(f"[WARN] Retrieval detail failed for {rid}: {e}")

            context["retrieval_detailed"] = detailed

        # Image and attention maps, ig maps, hybrid maps
        context["img_orig"]   = np_to_base64_img(img_tensor.squeeze().cpu().numpy(), cmap="gray")
        context["attn_map"]   = np_to_base64_img(out["attention_map"], cmap="jet")
        top1 = context["topk_idx"][0]
        context["ig_map"]     = np_to_base64_img(out["ig_maps"][top1], cmap="hot")
        hybrid = np.stack([
            out["attention_map"],
            np.zeros_like(out["attention_map"]),
            out["ig_maps"][top1]
        ], axis=-1)
        context["hybrid_map"] = np_to_base64_img(hybrid)

    print("Context keys:", list(context.keys()))
    print("Retrieval pairs:", context.get("retrieval"))

    return render_template("index.html", **context)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
