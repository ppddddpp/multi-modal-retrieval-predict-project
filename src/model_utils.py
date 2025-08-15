from transformers import AutoModel, AutoTokenizer
from pathlib import Path

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

MODEL_PLACE = BASE_DIR / "models"

def load_hf_model_or_local(
    model_name: str,
    local_dir: (str | Path) = None,
    is_tokenizer: bool = False,
    **kwargs
):
    """
    Load a Hugging Face model or tokenizer from a local directory if valid, 
    otherwise download from HF hub, save locally, and return it.

    Args:
        model_name (str): HF model/tokenizer identifier.
        local_dir (str | Path, optional): Directory to load from or save to.
                                           Defaults to MODEL_PLACE.
        is_tokenizer (bool): If True, loads a tokenizer; else loads a model.
        **kwargs: Passed through to `from_pretrained`.

    Returns:
        Pretrained model or tokenizer.
    """
    local_dir = Path(local_dir) if local_dir else MODEL_PLACE
    loader = AutoTokenizer.from_pretrained if is_tokenizer else AutoModel.from_pretrained

    # Attempt local load first
    try:
        print(f"[Local] Trying to load {'tokenizer' if is_tokenizer else 'model'} from {local_dir}")
        return loader(str(local_dir), **kwargs)
    except (OSError, ValueError):
        # Fall through to download if local load fails
        print(f"[Download] {model_name} to will save to {local_dir}")

    # Download from HF hub and save
    local_dir.mkdir(parents=True, exist_ok=True)
    instance = loader(model_name, cache_dir=str(local_dir), **kwargs)
    instance.save_pretrained(str(local_dir))
    return instance
