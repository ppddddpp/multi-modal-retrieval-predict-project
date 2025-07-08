from transformers import AutoModel, AutoTokenizer
from pathlib import Path

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

MODEL_PLACE = BASE_DIR / 'models'

def load_hf_model_or_local(model_name, local_dir=None, is_tokenizer=False, **kwargs):
    """
    Load a Hugging Face model or tokenizer from a local directory if it exists, else download and save to that directory.

    Args:
        model_name (str): The name of the model or tokenizer to load.
        local_dir (str, Path): The local directory to look for the model or tokenizer, or None to use
            `MODEL_PLACE` from `model_utils.py`.
        is_tokenizer (bool): Whether to load a tokenizer or not. Defaults to False.
        **kwargs: Any additional keyword arguments to pass to the Hugging Face `from_pretrained` method.

    Returns:
        The loaded model or tokenizer.
    """
    from_pretrained = AutoTokenizer.from_pretrained if is_tokenizer else AutoModel.from_pretrained

    if local_dir:
        local_dir = Path(local_dir)
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"[Local] Loading {'tokenizer' if is_tokenizer else 'model'} from: {local_dir}")
            return from_pretrained(str(local_dir), **kwargs)
        else:
            print(f"[Download] {model_name} → Saving to {local_dir}")
            local_dir.mkdir(parents=True, exist_ok=True)
            model = from_pretrained(model_name, cache_dir=str(local_dir), **kwargs)
            model.save_pretrained(str(local_dir))
            if is_tokenizer:
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(local_dir), **kwargs)
                tokenizer.save_pretrained(str(local_dir))
                return tokenizer
            return model
    else:
        local_dir = MODEL_PLACE
        print(f"[Download] {model_name} → Saving to {local_dir}")
        local_dir.mkdir(parents=True, exist_ok=True)
        model = from_pretrained(model_name, cache_dir=str(local_dir), **kwargs)
        model.save_pretrained(str(local_dir))
        if is_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(local_dir), **kwargs)
            tokenizer.save_pretrained(str(local_dir))
            return tokenizer
        return model
