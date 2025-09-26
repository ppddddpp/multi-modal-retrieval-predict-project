from scipy.stats import pearsonr, spearmanr
import numpy as np
import io
import re
import base64
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
from PIL import Image
import math
from typing import Optional, Dict, Any
from pathlib import Path
import pickle
from sklearn.metrics import roc_auc_score

from Helpers import Config
from DataHandler import  parse_openi_xml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CKPT_DIR = BASE_DIR / "checkpoints"
MODEL_DIR = BASE_DIR / "models"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
OUTPUT_DIR = BASE_DIR / "outputs"
XML_DIR = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
CONFIG_DIR = BASE_DIR / "configs"
CKPT_PATH = BASE_DIR / "checkpoints" / "model_best.pt"
REPORT_CACHE_PATH = OUTPUT_DIR / "openi_reports.pkl"

cfg = Config.load(CONFIG_DIR / "config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[loader] device = {device}")

def load_report_lookup_via_parser(xml_dir=None, dicom_root=None,
                                    report_cache_path=None,
                                    combined_groups=None
                                ) -> dict:
    """Load a mapping of OpenI id -> report_text from a cache file (if present)
    or by parsing XML files in the given directory.

    Args:
        xml_dir: Directory containing the OpenI XML files.
        dicom_root: Directory containing the OpenI DICOM files.

    Returns:
        A dictionary mapping OpenI id to report_text.
    """
    if xml_dir is None:
        xml_dir = XML_DIR
    if dicom_root is None:
        dicom_root = DICOM_ROOT
    if report_cache_path is None:
        report_cache_path = REPORT_CACHE_PATH
    
    if report_cache_path.exists():
        print(f"[loader] Loading cached report lookup from {report_cache_path.name}")
        with open(report_cache_path, "rb") as f:
            return pickle.load(f)

    print("[loader] Parsing reports using parse_openi_xml().")
    parsed_records = parse_openi_xml(xml_dir=xml_dir, dicom_root=dicom_root, combined_groups=combined_groups)
    id_to_report = {
        rec["id"]: rec["report_text"]
        for rec in parsed_records
        if "id" in rec and "report_text" in rec
    }

    report_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_cache_path, "wb") as f:
        pickle.dump(id_to_report, f)
    print(f"[loader] Cached {len(id_to_report)} reports to {report_cache_path.name}")
    return id_to_report

def find_dicom_file(rid: str, dicom_root=None) -> Path:
    """Find the DICOM file path for the given id.

    Args:
        rid: The id of the DICOM file to find.
        dicom_root: The root directory of the DICOM files.

    Returns:
        The path to the DICOM file.
    """
    if dicom_root is None:
        dicom_root = DICOM_ROOT
    
    primary = list(dicom_root.rglob(f"{rid}.dcm"))
    if primary:
        return primary[0]

    # Try fallback without leading patient ID
    fallback_id = "_".join(rid.split("_")[1:])
    fallback = list(dicom_root.rglob(f"{fallback_id}.dcm"))
    if fallback:
        print(f"[loader] Using fallback DICOM path: {fallback[0].name}")
        return fallback[0]

    raise FileNotFoundError(f"DICOM not found for either {rid}.dcm or {fallback_id}.dcm")

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

def resize_to_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Resize src (H_src x W_src) to match shape of ref (H_ref x W_ref).
    Returns resized array (float32). Tries cv2, then skimage, then scipy.ndimage.zoom,
    then a crude numpy repeat fallback.
    """
    import numpy as np
    Ht, Wt = ref.shape
    Hs, Ws = src.shape
    if (Hs, Ws) == (Ht, Wt):
        return src.astype(np.float32)

    src_f = src.astype(np.float32)

    # try cv2
    try:
        import cv2
        # cv2.resize takes (width, height)
        resized = cv2.resize(src_f, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        return resized.astype(np.float32)
    except Exception:
        pass

    # try skimage
    try:
        from skimage.transform import resize
        resized = resize(src_f, (Ht, Wt), order=1, preserve_range=True, anti_aliasing=True)
        return resized.astype(np.float32)
    except Exception:
        pass

    # try scipy.ndimage.zoom
    try:
        from scipy.ndimage import zoom
        zh = Ht / float(Hs)
        zw = Wt / float(Ws)
        resized = zoom(src_f, (zh, zw), order=1)  # order=1 -> bilinear-like
        return resized.astype(np.float32)
    except Exception:
        pass

    # fallback: crude nearest-repeat
    try:
        # compute integer repetition factors (ceil), then crop
        rh = int(np.ceil(Ht / Hs))
        rw = int(np.ceil(Wt / Ws))
        rep = np.repeat(np.repeat(src_f, rh, axis=0), rw, axis=1)
        resized = rep[:Ht, :Wt].astype(np.float32)
        return resized
    except Exception:
        # last resort: pad or crop
        out = np.zeros((Ht, Wt), dtype=np.float32)
        h = min(Hs, Ht)
        w = min(Ws, Wt)
        out[:h, :w] = src_f[:h, :w]
        return out

def compare_maps(map_a: np.ndarray, map_b: np.ndarray, topk_frac: float = 0.05):
    """
    Compute metrics comparing two attention maps.
    """
    a = map_a.flatten()
    b = map_b.flatten()

    # Defaults
    pearson_val = 0.0
    spearman_val = 0.0

    try:
        a_const = np.all(a == a[0])
        b_const = np.all(b == b[0])

        if a_const or b_const:
            const_source = "a" if a_const else "b"
            print(f"[WARN] compare_maps: input {const_source} is constant, correlation undefined")
        else:
            pearson_val = float(pearsonr(a, b)[0])
            spearman_val = float(spearmanr(a, b).correlation)
    except Exception as e:
        print(f"[WARN] compare_maps correlation failed: {e}")

    # IoU on top-k fraction
    k = max(1, int(len(a) * topk_frac))
    a_top = (a >= np.partition(a, -k)[-k])
    b_top = (b >= np.partition(b, -k)[-k])
    inter = np.logical_and(a_top, b_top).sum()
    union = np.logical_or(a_top, b_top).sum()
    iou = float(inter) / (float(union) + 1e-8)

    return {
        'pearson': pearson_val,
        'spearman': spearman_val,
        f'iou_top{int(topk_frac*100)}pct': iou
    }

def to_numpy(x):
    """Return a numpy array whether x is a torch tensor or numpy already."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def heatmap_to_base64_overlay(orig_img: np.ndarray,
                            heatmap: np.ndarray,
                            cmap: str = 'jet',
                            alpha: float = 0.5) -> str:
    # convert orig to 2D grayscale if needed
    """
    Returns a base64-encoded PNG of the original image with the heatmap overlayed.

    Parameters
    ----------
    orig_img : np.ndarray
        The original image. May be 2D (grayscale) or 3D (RGB).
    heatmap : np.ndarray
        The heatmap. Will be resized if shape is different from `orig_img`.
    cmap : str, optional
        The matplotlib colormap to use. Defaults to 'jet'.
    alpha : float, optional
        The alpha channel value of the heatmap. Defaults to 0.5.

    Returns
    -------
    str
        The base64-encoded PNG.
    """
    img = orig_img.squeeze()
    if img.ndim == 3 and img.shape[2] == 3:
        base = img
    else:
        # grayscale -> RGB
        if img.max() <= 1.0:
            img_u8 = (img * 255).astype('uint8')
        else:
            img_u8 = img.astype('uint8')
        base = np.stack([img_u8, img_u8, img_u8], axis=-1)
        base = base.astype('uint8')

    # normalize heatmap to 0..1
    h = heatmap.astype(np.float32)
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)

    # resize heatmap if different shape
    H, W = base.shape[:2]
    if h.shape != (H, W):
        h_img = Image.fromarray((h * 255).astype('uint8')).resize((W, H), resample=Image.BILINEAR)
        h = np.asarray(h_img).astype(np.float32) / 255.0

    # colorize via matplotlib cmap
    cmap_fn = plt.get_cmap(cmap)
    colored = cmap_fn(h)[:, :, :3]  # HxWx3 float 0..1
    colored_u8 = (colored * 255).astype('uint8')

    # ensure base is uint8
    if base.dtype != np.uint8:
        if base.max() <= 1.0:
            base_u8 = (base * 255).astype('uint8')
        else:
            base_u8 = base.astype('uint8')
    else:
        base_u8 = base

    # blend
    blended = (base_u8 * (1 - alpha) + colored_u8 * alpha).astype('uint8')

    pil = Image.fromarray(blended)
    buf = io.BytesIO()
    pil.save(buf, format='PNG', optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def save_b64_map(
    b64_str: str,
    qid: str,
    rid: str = None,
    map_type: str = "attn_txt",
    base_dir: str = "outputs/reports"
) -> str:
    """
    Save a base64-encoded map into the folder of the query (qid).

    Parameters
    ----------
    b64_str : str
        Base64 string (from heatmap_to_base64_overlay).
    qid : str
        Query ID.
    rid : str, optional
        Retrieval ID. If None, saves under 'query' folder.
    map_type : str
        Type of map: 'attn_txt', 'attn_img', 'attn_comb', 'ig', 'gradcam'.
    base_dir : str
        Base output directory. Defaults to 'outputs/reports'.

    Returns
    -------
    str
        Path to saved PNG.
    """
    if b64_str is None:
        return None

    img_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_data))

    if rid is None:
        out_dir = Path(base_dir) / str(qid) / "query"
        fname = f"{map_type}.png"
    else:
        out_dir = Path(base_dir) / str(qid) / str(rid)
        fname = f"{map_type}.png"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    img.save(out_path, format="PNG", optimize=True)
    return str(out_path)

def attention_to_html(tokens, scores):
    """
    Convert attention scores to HTML visualization.

    Parameters
    ----------
    tokens : list[str]
        Tokens to visualize.
    scores : np.ndarray
        Attention scores for each token.

    Returns
    -------
    str
        HTML visualization of attention scores.
    """
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    spans = []
    for tok, s in zip(tokens, scores):
        color = f"rgba(255,0,0,{s:.2f})"  # red with alpha = score
        spans.append(f"<span style='background-color:{color}'>{tok}</span>")
    return " ".join(spans)

def make_attention_maps(
    model,
    fusion_layer,                 # the fusion module (callable) that supports return_attention=True
    img_global,                   # tensor (B,...) single-batch or (1,...)
    img_patches,                  # tensor (B,...) single-batch or (1,...)
    txt_feats,                    # tensor (B,...) single-batch or (1,...)
    device,
    rid: Optional[str] = None,
    combine_method: str = "avg",  # "avg" | "product" | "max"
    pad_to_square: bool = True,   # pad attention vector to next perfect square if needed
    save: bool = False,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute attention maps for a single input using the given fusion layer and explainer.

    Parameters
    ----------
    model : Module
        The model containing the explainer and fusion layer.
    fusion_layer : callable
        The fusion module that supports `return_attention=True`.
    img_global : Tensor
        The global image features. Expected shape: (B, ...) where B is the batch size.
    img_patches : Tensor
        The image patches. Expected shape: (B, ...) where B is the batch size.
    txt_feats : Tensor
        The text features. Expected shape: (B, ...) where B is the batch size.
    device : torch.device
        The device to run the computation on.
    rid : Optional[str]
        The request ID (if any) for logging or saving.
    combine_method : str
        The method to combine the text->image and image->text attention maps. One of "avg", "product", or "max".
        Defaults to "avg".
    pad_to_square : bool
        Whether to pad the attention vector to the next perfect square if needed. Defaults to True.
    save : bool
        Whether to save the attention maps to disk. Defaults to False.
    save_dir : Optional[str]
        The directory to save the attention maps to. Defaults to the `BASE_DIR/attention_maps/eval` directory.

    Returns
    -------
    out : Dict[str, Any]
        A dictionary containing the attention maps, tensors, and other information.
        The keys are:
            - "att_map_txt": The text->image attention map.
            - "att_map_img": The image->text attention map.
            - "att_map_comb": The combined attention map.
            - "att_txt_tensor": The text->image attention tensor.
            - "att_img_tensor": The image->text attention tensor.
            - "att_comb_tensor": The combined attention tensor.
    """
    out = {
        "att_map_txt": None,
        "att_map_img": None,
        "att_map_comb": None,
        "att_txt_tensor": None,
        "att_img_tensor": None,
        "att_comb_tensor": None,
    }

    try:
        if fusion_layer is None:
            raise RuntimeError("fusion_layer is None")

        # Make sure inputs are 1-batch and on device
        ig = img_global[0:1].to(device)
        ip = img_patches[0:1].to(device)
        tf = txt_feats[0:1].to(device)

        # request attention dict
        _, att = fusion_layer(ig, ip, tf, return_attention=True)
        if att is None:
            raise RuntimeError("fusion returned no attention")

        # helper: coerce common shapes to (B,1,N)
        def _to_patch_attention(a):
            if a is None:
                return None
            if not torch.is_tensor(a):
                try:
                    a = torch.as_tensor(a, device=device)
                except Exception:
                    a = torch.tensor(a, device=device)
            # common cases:
            # (B, heads, N) -> mean over heads -> (B,1,N)
            # (B,1,N) -> keep
            # (B,N,1) -> transpose -> (B,1,N)
            if a.dim() == 3:
                B, A, C = a.shape
                # already (B,1,N)
                if A == 1:
                    return a
                # (B,heads,N) -> average heads
                if C > 1:
                    return a.mean(dim=1, keepdim=True)
                # (B,N,1) -> transpose to (B,1,N)
                if C == 1:
                    return a.transpose(1, 2)
            if a.dim() == 2:
                # (B,N) -> (B,1,N)
                return a.unsqueeze(1)
            # otherwise return as-is
            return a

        att_txt = _to_patch_attention(att.get("txt2img", None))
        att_img = _to_patch_attention(att.get("img2txt", None))

        if att_txt is None and att_img is None:
            raise RuntimeError("no usable txt2img or img2txt attention found")

        # align patch counts if both exist
        if att_txt is not None and att_img is not None and att_txt.shape[-1] != att_img.shape[-1]:
            # try simple heuristics; otherwise prefer txt2img
            n_txt = att_txt.shape[-1]
            n_img = att_img.shape[-1]
            if n_img == 1 and n_txt > 1:
                att_img = att_img.expand(-1, -1, n_txt)
            elif n_txt == 1 and n_img > 1:
                att_txt = att_txt.expand(-1, -1, n_img)
            else:
                # fallback: drop img->txt (prefer text->image mapping)
                att_img = None

        # choose combination
        if att_txt is None:
            comb = att_img
        elif att_img is None:
            comb = att_txt
        else:
            if combine_method == "product":
                comb = att_txt * att_img
            elif combine_method == "max":
                comb = torch.max(att_txt, att_img)
            else:  # avg
                comb = (att_txt + att_img) / 2.0

        # normalize comb along patches
        comb = comb.float()
        denom = comb.sum(dim=-1, keepdim=True)
        comb = comb / (denom + 1e-8)

        # store raw tensors
        out["att_txt_tensor"] = att_txt
        out["att_img_tensor"] = att_img
        out["att_comb_tensor"] = comb

        # helper to prepare grid size, pad if required, and call explainer
        def _prepare_and_map(att_tensor):
            if att_tensor is None:
                return None
            B = att_tensor.shape[0]
            N = int(att_tensor.shape[-1])
            G = int(math.isqrt(N))
            if G * G != N:
                # try ceil to next square
                G = int(math.ceil(math.sqrt(N)))
                target_N = G * G
                if pad_to_square and target_N >= N:
                    # pad zeros on last dim to match target_N
                    pad_len = target_N - N
                    pad_tensor = torch.zeros((B, 1, pad_len), device=att_tensor.device, dtype=att_tensor.dtype)
                    att_tensor = torch.cat([att_tensor, pad_tensor], dim=-1)
                else:
                    # unable to reshape reasonably
                    print(f"[WARN] attention length N={N} is not a perfect square and pad_to_square is False")
                    return None
            # now call explainer
            try:
                return model.explainer.compute_attention_map(att_tensor, grid_size=G)
            except Exception as e:
                print(f"[WARN] compute_attention_map failed for rid={rid}: {e}")
                return None

        out["att_map_txt"] = _prepare_and_map(att_txt)
        out["att_map_img"] = _prepare_and_map(att_img)
        out["att_map_comb"] = _prepare_and_map(comb)

        # optional save
        if save:
            if save_dir is None:
                save_dir = Path(BASE_DIR) / "attention_maps" / "eval"

            os.makedirs(save_dir, exist_ok=True)
            def _save_map(m, fname):
                if m is None:
                    return
                arr = np.array(m).astype(np.float32)
                # normalize to 0..1
                if (arr.max() - arr.min()) > 1e-8:
                    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                else:
                    arr = np.clip(arr, 0.0, 1.0)
                img = Image.fromarray((arr * 255).astype(np.uint8))
                img.save(os.path.join(save_dir, fname))

            _save_map(out["att_map_txt"], f"{rid}_txt2img.png")
            _save_map(out["att_map_img"], f"{rid}_img2txt.png")
            _save_map(out["att_map_comb"], f"{rid}_combined_{combine_method}.png")

        return out

    except Exception as e:
        print(f"[WARN] make_attention_maps failed for {rid}: {e}")
        # ensure returned structure exists even on error
        return out

def kg_alignment_loss(joint_emb, batch_ids, kg_embs, node2id, trainer, labels=None, label_cols=None, loss_type="cosine"):
    """
    Align joint embeddings with KG embeddings.
    - First tries explicit 'report:{id}' node
    - If missing, falls back to average of its label-based KG nodes
    - If no mapping exists, falls back to zero vector
    """
    if joint_emb.device != kg_embs.device:
        kg_embs = kg_embs.to(joint_emb.device)

    kg_vecs = []
    for i, id_ in enumerate(batch_ids):
        node_key = f"report:{id_}"
        if node_key in node2id:
            # direct mapping
            kg_vecs.append(kg_embs[node2id[node_key]])
        else:
            # try label-based fallback
            if labels is not None and label_cols is not None:
                if i < len(labels):  # safety check
                    label_vec = labels[i].cpu().numpy()
                    pos_labels = [label_cols[j] for j, v in enumerate(label_vec) if v > 0.5]

                    label_embs = []
                    for lab in pos_labels:
                        lab_key = f"label:{lab}"
                        if lab_key in node2id:
                            label_embs.append(kg_embs[node2id[lab_key]])

                    if len(label_embs) > 0:
                        kg_vecs.append(torch.stack(label_embs).mean(dim=0))
                        continue  # done with fallback

            # if no report node and no labels mapped
            kg_vecs.append(torch.zeros_like(kg_embs[0]))

    kg_vecs = torch.stack(kg_vecs).to(joint_emb.device)

    # project joint_emb into KG space
    joint_proj = trainer.proj_to_kg(joint_emb)

    if loss_type == "mse":
        return torch.nn.functional.mse_loss(joint_proj, kg_vecs)
    elif loss_type == "cosine":
        return 1 - torch.nn.functional.cosine_similarity(joint_proj, kg_vecs).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

def log_and_print(*msgs, log_file=None):
    text = " ".join(str(m) for m in msgs)
    print(text)
    
    if log_file is None:
        raise ValueError("log_file must be provided")
    
    log_file = Path(log_file)
    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def _sanitize_node(s: str) -> str:
    s2 = s.strip()
    s2 = re.sub(r'\s+', '_', s2)
    s2 = re.sub(r'[^A-Za-z0-9_:.-]', '', s2)
    return s2

def safe_roc_auc(y_true, y_pred, label_names=None):
    results = []
    skipped = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            results.append(float("nan"))
            if label_names is not None:
                skipped.append(label_names[i])
        else:
            results.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    if skipped:
        print(f"Skipped AUROC for classes with no positives/negatives: {skipped}")
    return np.array(results)

__all__ = ("find_dicom_file", "load_report_lookup_via_parser",
            "report_lookup", "make_attention_maps", "attention_to_html", "kg_alignment_loss", "log_and_print", "_sanitize_node", "safe_roc_auc")
