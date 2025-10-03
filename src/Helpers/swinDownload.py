import timm
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PLACE = BASE_DIR / 'models'
MODEL_PLACE.mkdir(exist_ok=True)

def safe_transfer(src, dst, method="move"):
    try:
        if method == "copy":
            print(f"Copying {src} to {dst}")
            shutil.copy2(src, dst)
        else:
            print(f"Moving {src} to {dst}")
            shutil.move(src, dst)
    except OSError:
        print("[WARN] Move failed, falling back to copy...")
        shutil.copy2(src, dst)

def download_swin(swin_name=None, swin_ckpt_path=None, method="move", swin_channels=3):
    """
    Download a Swin Transformer checkpoint and save to a target location.

    Parameters
    ----------
    swin_name : str, optional
        Timm model name (default = 'swin_base_patch4_window7_224')
    swin_ckpt_path : str or Path, optional
        Where to save the .safetensors checkpoint.
        If None, defaults to BASE_DIR/models/swin_checkpoint.safetensors
    method : {'copy','move'}, optional
        Whether to copy or move the file (default = move)
    swin_channels : int, optional
        Number of channels in input images (default = 3)

    Returns
    -------
    Path
        The path to the downloaded Swin checkpoint.
    """
    model_name = 'swin_base_patch4_window7_224' if swin_name is None else swin_name
    print(f"Triggering download for {model_name}...")
    # Use 3 channels since your preprocessor now outputs RGB
    _ = timm.create_model(model_name, pretrained=True, in_chans=swin_channels)

    # Find cached safetensors
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_root = next(hf_cache_dir.glob(f"models--timm--{model_name.replace('/', '--')}*"))
    model_file = next(model_cache_root.rglob("model.safetensors"))

    # Decide target
    if swin_ckpt_path is None:
        swin_ckpt_path = MODEL_PLACE / "swin_checkpoint.safetensors"
    else:
        swin_ckpt_path = Path(swin_ckpt_path)
        swin_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    safe_transfer(model_file, swin_ckpt_path, method)
    print(f"Swin checkpoint saved at: {swin_ckpt_path.resolve()}")
    return swin_ckpt_path

if __name__ == "__main__":
    download_swin()