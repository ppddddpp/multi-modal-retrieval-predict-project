from tensorDICOM import DICOMImagePreprocessor
from transformers import AutoTokenizer
from ChestXRDataset import ChestXRDataset
from torch.utils.data import DataLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PLACE = BASE_DIR / "models"

def tokenize_report(text, tokenizer, max_length=128):
    """
    Tokenize report using Hugging Face tokenizer.

    Args:
        text (str): Report text
        tokenizer (transformers.AutoTokenizer): Hugging Face tokenizer
        max_length (int, optional): Tokenization length. Defaults to 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (input_ids, attention_mask)
    """
    tokens = tokenizer(
        text or "",
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)

def build_dataloader(records, batch_size=4, shuffle=True, num_workers=4,
                     mean=0.5, std=0.5, tokenizer=None):
    """
    Convenience function to create DataLoader for ChestXRDataset.

    Args:
        records (List[dict]): List of records parsed from OpenI XML.
        batch_size (int, optional): Defaults to 4.
        shuffle (bool, optional): Defaults to True.
        num_workers (int, optional): Defaults to 4.
        mean (float, optional): Image normalization mean. Defaults to 0.5.
        std (float, optional): Image normalization std. Defaults to 0.5.
        tokenizer (transformers.AutoTokenizer, optional): Defaults to None.

    Returns:
        torch.utils.data.DataLoader
    """
    preprocessor = DICOMImagePreprocessor(mean=mean, std=std)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            cache_dir=str(MODEL_PLACE / "clinicalbert")
        )
    dataset = ChestXRDataset(records, image_preprocessor=preprocessor,
                             tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)
