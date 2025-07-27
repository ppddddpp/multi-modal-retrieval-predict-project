import io
import base64
from pathlib import Path

from flask import Flask, render_template, request
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import Config
from src.tensorDICOM import DICOMImagePreprocessor
from src.model import MultiModalRetrievalModel
from src.labeledData import disease_groups, normal_groups
from transformers import AutoTokenizer

# ── Project directories ────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
CKPT_DIR       = BASE_DIR / "checkpoints"
MODEL_DIR      = BASE_DIR / "models"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
CONFIG_DIR     = BASE_DIR / "config"
# ────────────────────────────────────────────────────────────────────────────────

cfg = Config.load(CONFIG_DIR / "config.yaml")
app = Flask(__name__, static_folder="static", template_folder="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_cols = list(disease_groups.keys()) + list(normal_groups.keys())

model = MultiModalRetrievalModel(
    joint_dim=cfg.joint_dim,      
    num_heads=cfg.num_heads,        
    num_classes=len(label_cols),
    fusion_type=cfg.fusion_type,
    swin_ckpt_path=str(MODEL_DIR / "swin_base_patch4_window7_224.pth"),
    bert_local_dir=str(MODEL_DIR / "Bio_ClinicalBERT")
).to(device)

state = torch.load(CKPT_DIR / "model_best.pt", map_location=device)
model.load_state_dict(state)
model.eval()

preproc   = DICOMImagePreprocessor()
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def np_to_base64_img(arr: np.ndarray, cmap="gray") -> str:
    fig, ax = plt.subplots(figsize=(4,4))
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
        # 1) DICOM + text
        dcm_bytes  = request.files["dicom_file"].read()
        img_tensor = preproc(dcm_bytes).unsqueeze(0).to(device)

        tokens    = tokenizer(
            request.form["text_query"],
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        txt_ids  = tokens.input_ids.to(device)
        txt_mask = tokens.attention_mask.to(device)

        # 2) Predict + explain
        out = model.predict(img_tensor, txt_ids, txt_mask, K=5, explain=True)

        # 3) Assemble context
        # Classification top‐K
        context["topk_idx"]  = out["topk_idx"][0]
        context["topk_vals"] = out["topk_vals"][0]

        # Retrieval
        context["retrieval"] = list(zip(out["retrieval_ids"], out["retrieval_dists"]))

        # Original image
        context["img_orig"]  = np_to_base64_img(img_tensor.squeeze().cpu().numpy(), cmap="gray")

        # Attention map
        context["attn_map"]  = np_to_base64_img(out["attention_map"], cmap="jet")

        # IG map for the top‐1 predicted class
        top1 = context["topk_idx"][0]
        context["ig_map"]    = np_to_base64_img(out["ig_maps"][top1], cmap="hot")

        # Hybrid
        hybrid = np.stack([
            out["attention_map"],
            np.zeros_like(out["attention_map"]),
            out["ig_maps"][top1]
        ], axis=-1)
        context["hybrid_map"] = np_to_base64_img(hybrid)

    return render_template("index.html", **context)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
