import timm
import torch
import os
from pathlib import Path
from safetensors.torch import save_file as safetensors_save

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PLACE = BASE_DIR / 'models'
MODEL_PLACE.mkdir(exist_ok=True)

def adjust_in_chans(state_dict, desired_in_chans):
    """Adapt patch_embed.proj.weight to desired input channels (in-memory)."""
    key = "patch_embed.proj.weight"
    if key not in state_dict:
        return state_dict  # nothing to fix

    w = state_dict[key]
    # ensure tensor
    if not isinstance(w, torch.Tensor):
        w = torch.as_tensor(w)

    if w.shape[1] == desired_in_chans:
        return state_dict  # already matches

    if w.shape[1] == 1 and desired_in_chans == 3:
        # repeat gray channel 3x
        new_w = w.repeat(1, 3, 1, 1).contiguous()
    elif w.shape[1] == 3 and desired_in_chans == 1:
        # average rgb into gray
        new_w = w.mean(dim=1, keepdim=True).contiguous()
    else:
        # generic slice/expand
        if w.shape[1] < desired_in_chans:
            reps = (1, desired_in_chans // w.shape[1] + 1, 1, 1)
            new_w = w.repeat(reps)[:, :desired_in_chans, :, :].contiguous()
        else:
            new_w = w[:, :desired_in_chans, :, :].contiguous()

    state_dict[key] = new_w
    return state_dict

def download_swin(swin_name=None, swin_ckpt_path=None, swin_channels=3, tmp_when_overwrite=True):
    """
    Download a Swin Transformer checkpoint and save to safetensors.

    - swin_channels: desired input channels (1 or 3).
    - tmp_when_overwrite: when True, save to a *.tmp file and then atomically attempt replace.
    Returns the Path to the saved checkpoint.
    """
    if swin_channels not in (1, 3):
        print(f"[WARN] unusual swin_channels={swin_channels}; function will still attempt to adapt.")

    model_name = 'swin_base_patch4_window7_224' if swin_name is None else swin_name
    print(f"Downloading {model_name} (base RGB weights) and adapting to in_chans={swin_channels}...")

    # Load RGB pretrained weights from timm (pretrained True downloads the official RGB weights)
    model = timm.create_model(model_name, pretrained=True, in_chans=3)
    state = model.state_dict()

    # Adjust first conv if needed
    if swin_channels != 3:
        state = adjust_in_chans(state, swin_channels)

    # Decide path
    if swin_ckpt_path is None:
        swin_ckpt_path = MODEL_PLACE / "swin_checkpoint.safetensors"
    else:
        swin_ckpt_path = Path(swin_ckpt_path)
        swin_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to tmp file first to avoid overwrite issues on Windows if the original is memory-mapped
    target_path = Path(swin_ckpt_path)
    write_path = target_path
    if tmp_when_overwrite and target_path.exists():
        write_path = target_path.with_suffix(target_path.suffix + ".tmp")

    # Ensure all tensors on CPU and plain torch.Tensor
    cpu_state = {str(k): (v.cpu() if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in state.items()}

    # Try safetensors write, fallback to torch.save if it fails
    try:
        safetensors_save(cpu_state, str(write_path))
    except Exception as e:
        print("[WARN] safetensors_save failed, falling back to torch.save:", e)
        try:
            torch.save(cpu_state, str(write_path))
        except Exception as e2:
            raise RuntimeError("Failed to save checkpoint via safetensors or torch.save") from e2

    # if we wrote to tmp, try to replace original atomically (ignore failure)
    if write_path != target_path:
        try:
            write_path.replace(target_path)
        except Exception as e:
            print("[INFO] Could not replace original file (ignored):", e)
            # keep tmp file as the saved checkpoint and return it
            return write_path

    print(f"Swin checkpoint saved at {target_path.resolve()}")
    return target_path


if __name__ == "__main__":
    download_swin()
