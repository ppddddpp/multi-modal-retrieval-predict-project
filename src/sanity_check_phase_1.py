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

# Resolve paths...
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

XML_DIR    = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
MODEL_PLACE = BASE_DIR / "models"
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_PLACE)

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

    # Parse records
    records = parse_openi_xml(str(XML_DIR), str(DICOM_ROOT))
    print("Loaded records:", len(records))

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
    img_feats, txt_feats = backbones(imgs, ids, masks)
    print("Image feats:", img_feats.shape, " NaNs?", torch.isnan(img_feats).any())
    print("Text feats: ", txt_feats.shape, " NaNs?", torch.isnan(txt_feats).any())

    # Embedding norms
    print("Img norm mean:", img_feats.norm(dim=1).mean().item())
    print("Txt norm mean:", txt_feats.norm(dim=1).mean().item())

    # Debug DICOM
    plot_dicom_debug(records[0]['dicom_path'])

    # Debug DICOM ranges
    B = imgs.size(0)
    dp = DICOMImagePreprocessor(augment=False)
    for i in range(B):
        arr = dp.load_raw_array(records[i]['dicom_path'])
        print(f"[WIN] img {i}  min={arr.min():.4f}, max={arr.max():.4f}")
        plt.imshow(arr, cmap='gray', vmin=arr.min(), vmax=arr.max())
        plt.axis('off')
        plt.show()