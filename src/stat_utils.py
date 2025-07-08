from tensorDICOM import DICOMImagePreprocessor
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np

class RawStatDataset(Dataset):
    def __init__(self, records, size=(224, 224)):
        self.records = records
        self.size = size  # target (width, height)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            # Load raw DICOM image
            pre = DICOMImagePreprocessor(augment=False)
            arr = pre.load_raw_array(rec['dicom_path'])  # shape: (H, W)

            # Resize to (H, W) = (224, 224)
            resized = cv2.resize(arr, self.size, interpolation=cv2.INTER_AREA)

            # Normalize to [0, 1]
            if resized.max() > 1:
                resized = resized.astype(np.float32) / 255.0

            # Convert to 3-channel when using Swin (by duplicating channels)
            resized = np.stack([resized] * 3, axis=0)  # shape: (3, 224, 224)

            return torch.from_numpy(resized).float()

        except Exception as e:
            print(f"[ERROR] Failed at idx={idx} → {rec['dicom_path']}: {e}")
            return torch.zeros((3, *self.size), dtype=torch.float32)  # fallback

