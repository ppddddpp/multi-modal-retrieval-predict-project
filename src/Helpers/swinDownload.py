import timm
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PLACE = BASE_DIR / 'models'
MODEL_PLACE.mkdir(exist_ok=True)

def download_swin(swin_name=None, swin_ckpt_path=None, method=None):
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
    """
    # Pick model name
    model_name = 'swin_base_patch4_window7_224' if swin_name is None else swin_name
    print(f"Triggering download for {model_name}...")
    _ = timm.create_model(model_name, pretrained=True, in_chans=1)

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

    # Copy or move
    if method == "copy":
        print(f"Copying {model_file} to {swin_ckpt_path}")
        shutil.copy2(model_file, swin_ckpt_path)
    else:
        print(f"Moving {model_file} to {swin_ckpt_path}")
        shutil.move(model_file, swin_ckpt_path)

    print(f"Swin checkpoint saved at: {swin_ckpt_path.resolve()}")
    return swin_ckpt_path

if __name__ == "__main__":
    download_swin()