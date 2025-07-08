import timm
import os
from pathlib import Path
import shutil

# Trigger download (uses HuggingFace under the hood)
model_name = 'swin_base_patch4_window7_224'
print(f"Triggering download for {model_name}...")
model = timm.create_model(model_name, pretrained=True, in_chans=1)

# Find the safetensors file from the Hugging Face cache
hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_cache_root = next(hf_cache_dir.glob(f"models--timm--{model_name.replace('/', '--')}*"))
model_file = next(model_cache_root.rglob("model.safetensors"))

# Copy or move it to your project
target_dir = Path("models")
target_dir.mkdir(exist_ok=True)
target_ckpt = target_dir / "swin_checkpoint.safetensors"

#print(f"Copying {model_file} to {target_ckpt}")
#shutil.copy2(model_file, target_ckpt)
#print(f"Swin checkpoint copied to: {target_ckpt.resolve()}")

print(f"Moving {model_file} to {target_ckpt}")
shutil.move(model_file, target_ckpt)
print(f"Swin checkpoint moved to: {target_ckpt.resolve()}")
