from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tensorDICOM import DICOMImagePreprocessor
import multiprocessing
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PLACE = BASE_DIR / "models"

def tokenize_report(text, tokenizer, max_length=128):
    """Tokenize report using Hugging Face tokenizer."""
    tokens = tokenizer(
        text or "",
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)

class ChestXRDataset(Dataset):
    def __init__(self, records, image_preprocessor=None, tokenizer=None, max_length=128):
        """
        Args:
            records (list of dicts): List of records with keys:
                - 'dicom_path'
                - 'report_text'
                - optional 'mesh_labels'
            image_preprocessor (DICOMImagePreprocessor, optional): Image preprocessor.
                Defaults to None.
            tokenizer (transformers.AutoTokenizer, optional): Tokenizer for text.
                Defaults to None.
            max_length (int, optional): Maximum length of text tokenization.
                Defaults to 128.
        """
        self.records = records
        self.image_preprocessor = image_preprocessor or DICOMImagePreprocessor()
        cache_path = str(MODEL_PLACE / "clinicalbert")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT",
                cache_dir=cache_path
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):    
        """
        Retrieve the dataset sample at the specified index.

        Args:
            idx (int): Index of the dataset sample to retrieve.

        Returns:
            dict: A dictionary containing the following keys:
                - 'image': Processed image tensor from the DICOM file.
                - 'input_ids': Tokenized input IDs of the report text.
                - 'attn_mask': Attention mask for the tokenized report text.
                - 'mesh_labels' (optional): Labels associated with the sample, if available.
        """
        rec = self.records[idx]
        # Process image
        img = self.image_preprocessor(rec['dicom_path'])        
        # Tokenize report
        input_ids, attn_mask = tokenize_report(
            rec.get('report_text', ''),
            self.tokenizer,
            self.max_length
        )
        sample = {
            'image': img,
            'input_ids': input_ids,
            'attn_mask': attn_mask
        }
        # Optional labels
        if 'mesh_labels' in rec:
            sample['mesh_labels'] = rec['mesh_labels']
        return sample
