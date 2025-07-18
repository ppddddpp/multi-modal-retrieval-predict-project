import os
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from model_utils import load_hf_model_or_local
from dataLoader import build_dataloader
from stat_utils import RawStatDataset
from dataParser import parse_openi_xml
from fusion import Backbones
from tensorDICOM import DICOMImagePreprocessor
import pydicom
import numpy as np
from collections import Counter
from labeledData import disease_groups, normal_groups

# Resolve paths...
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

XML_DIR    = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
MODEL_PLACE = BASE_DIR / "models"
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_PLACE)

def analyze_label_distribution(records,label_field='labels', label_names=None):
    """
    Count how often each label (from label vectors) appears in the dataset.
    
    Parameters
    ----------
    records : list of dicts
        Each record must have a 'labels' field that is a 14-dim list
    label_names : list of str
        List of names for the labels (optional but recommended)

    Returns
    -------
    Counter with label name (or index) as key and count as value
    """
    counter = Counter()
    for rec in records:
        vec = rec[label_field]
        for i, v in enumerate(vec):
            if v == 1:
                label = label_names[i] if label_names else i
                counter[label] += 1
    return counter

def plot_dicom_debug(dicom_path):
    """
    Plot debugging information for a single DICOM image.

    Plots the original raw image, the scaled image, the windowed image, and the
    normalized image. Prints out the min/max values of each.

    Args:
        dicom_path (str): path to the DICOM image file
    """
    dcm = pydicom.dcmread(dicom_path)
    raw = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, 'RescaleSlope', 1.0))
    intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
    scaled = raw * slope + intercept

    wc_val = dcm.get('WindowCenter', 40.0)
    ww_val = dcm.get('WindowWidth', 400.0)
    wc = float(wc_val[0] if isinstance(wc_val, pydicom.multival.MultiValue) else wc_val)
    ww = float(ww_val[0] if isinstance(ww_val, pydicom.multival.MultiValue) else ww_val)

    lower, upper = wc - ww / 2, wc + ww / 2
    win = np.clip(scaled, lower, upper)
    norm = (win - lower) / (upper - lower + 1e-5)

    print(f"[DEBUG] raw min/max       = {raw.min():.2f} / {raw.max():.2f}")
    print(f"        scaled min/max    = {scaled.min():.2f} / {scaled.max():.2f}")
    print(f"        window center/wid = {wc} / {ww}")
    print(f"        clip min/max      = {win.min():.2f} / {win.max():.2f}")
    print(f"        norm min/max      = {norm.min():.4f} / {norm.max():.4f}")

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    axs[0].hist(raw.ravel(), bins=100)
    axs[0].set_title("Raw")
    axs[1].hist(scaled.ravel(), bins=100)
    axs[1].set_title("Scaled")
    axs[2].hist(win.ravel(), bins=100)
    axs[2].set_title("Windowed")
    axs[3].hist(norm.ravel(), bins=100)
    axs[3].set_title("Normalized")
    plt.tight_layout()
    plt.show()

    plt.imshow(norm, cmap='gray', vmin=0, vmax=1)
    plt.title("Final Normalized Image")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Prepare a tokenizer for decoding
    print("XML DIR   exists:", XML_DIR.exists(), " →", XML_DIR)
    print("DICOM ROOT exists:", DICOM_ROOT.exists(), " →", DICOM_ROOT)
    tokenizer = load_hf_model_or_local("emilyalsentzer/Bio_ClinicalBERT", local_dir=MODEL_PLACE, is_tokenizer=True)

    combined_groups = {
    **disease_groups,
    **normal_groups
    }

    label_names = sorted(combined_groups.keys())
    normal_idx = label_names.index("Normal")

    # Parse records
    records = parse_openi_xml(str(XML_DIR), str(DICOM_ROOT))
    print("Loaded records:", len(records))

    for r in records:
        vec = r["labels"]
        r["is_normal"] = vec[normal_idx] == 1 and sum(vec) == 1
        r["is_abnormal"] = any(vec[i] for i in range(len(vec)) if i != normal_idx)

    label_counts = analyze_label_distribution(
        records,
        label_field='labels',
        label_names=label_names
    )

    print(f"→ Found {len(label_counts)} active labels:")
    for label, count in label_counts.most_common():
        print(f"  • {label:20s} — {count} cases")

    # Compute mean/std with RawStatDataset
    ds = RawStatDataset(records[:100])
    dl = DataLoader(ds, batch_size=16, num_workers=4, pin_memory=True)
    sum_, sum_sq, count = 0.0, 0.0, 0
    for batch in dl:
        b = batch.float()
        sum_   += b.sum().item()
        sum_sq += (b*b).sum().item()
        count  += b.numel()
    mean = sum_ / count
    std  = ((sum_sq/count) - mean**2)**0.5
    print(f"Computed mean={mean:.4f}, std={std:.4f}")

    # Build your multimodal DataLoader
    loader = build_dataloader(records, batch_size=4, mean=mean, std=std)

    # Get one batch
    batch = next(iter(loader))
    imgs, ids, masks = batch['image'], batch['input_ids'], batch['attn_mask']

    # Instantiate Backbones
    SWIN_CKPT = BASE_DIR / 'models' / 'swin_checkpoint.safetensors'
    BERT_DIR  = BASE_DIR / 'models' / 'clinicalbert_local/'
    backbones = Backbones(
        pretrained=True,
        swin_checkpoint_path=SWIN_CKPT,
        bert_local_dir=BERT_DIR
    )

    # Forward‐pass through Backbones
    (img_global, img_region), txt_feats = backbones(imgs, ids, masks)

    print("Global Image feats:", img_global.shape,  " NaNs?", torch.isnan(img_global).any())
    print("Patch Image feats: ", img_region.shape, " NaNs?", torch.isnan(img_region).any())
    print("Pooled Text feats: ", txt_feats.shape,  " NaNs?", torch.isnan(txt_feats).any())

    # See some norm mean 
    print("Global‑img norm mean:", img_global.norm(dim=1).mean().item())
    print("Patch‑img  norm mean:", img_region.norm(dim=2).mean().item())   # norm over C, then mean over patches&batch
    print("Text‑feat  norm mean:", txt_feats.norm(dim=1).mean().item())

    # Debug DICOM
    plot_dicom_debug(records[0]['dicom_path'])

    print("\n--- Report Text ---")
    print(records[0]['report_text'])
    print("\n--- Label Vector ---")
    print(records[0]['labels'])  
    print("Total records parsed:", len(records))
    total_label_hits = sum(sum(r['labels']) for r in records)
    print("Total labels across all records:", total_label_hits)
    avg_per_image = total_label_hits / len(records)
    print(f"Average labels per image: {avg_per_image:.2f}")
    n_normals = sum(r['is_normal'] for r in records)
    n_abnormals = sum(r['is_abnormal'] for r in records)
    print(f"Normal cases: {n_normals}")
    print(f"Abnormal cases: {n_abnormals}")

    # Debug DICOM ranges
    B = imgs.size(0)
    dp = DICOMImagePreprocessor(augment=False)
    for i in range(B):
        arr = dp.load_raw_array(records[i]['dicom_path'])
        print(f"[WIN] img {i}  min={arr.min():.4f}, max={arr.max():.4f}")
        plt.imshow(arr, cmap='gray', vmin=arr.min(), vmax=arr.max())
        plt.axis('off')
        plt.show()