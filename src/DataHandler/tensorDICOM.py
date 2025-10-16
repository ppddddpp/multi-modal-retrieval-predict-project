import io
import torch
import pydicom
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path

class DICOMImagePreprocessor:
    """
    Object-oriented DICOM loader and preprocessor.
    Applies windowing, normalization, and torchvision transforms.
    """
    def __init__(self, mean=0.5, std=0.5,
                    default_window_center=40.0, default_window_width=400.0,
                    output_size=(224, 224), augment=False):
        
        if isinstance(mean, (list, tuple, np.ndarray)):
            if len(mean) != 3:
                raise ValueError("mean must be scalar or length-3 iterable")
            mean3 = [float(x) for x in mean]
        else:
            mean3 = [float(mean)] * 3

        if isinstance(std, (list, tuple, np.ndarray)):
            if len(std) != 3:
                raise ValueError("std must be scalar or length-3 iterable")
            std3 = [float(x) for x in std]
        else:
            std3 = [float(std)] * 3

        self.mean = mean3
        self.std = std3
        self.default_center = default_window_center
        self.default_width = default_window_width
        # Define torchvision pipeline
        ops = [
            # transforms.ToPILImage(),       # For one channel uncomment this
            transforms.Resize(output_size)
        ]
        if augment:
            ops += [
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(0.1)
            ]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean3, std3)
        ]
        self.transform = transforms.Compose(ops)

    def window_image(self, pixel_array, window_center, window_width):
        """
        Apply windowing transformation to raw pixel data based on the specified
        window center and width. This operation maps the pixel values to a 
        normalized range [0, 1] by clipping the pixel array within the specified 
        window and then scaling it.

        Parameters:
        pixel_array (np.ndarray): Raw pixel data to be transformed.
        window_center (float): The center value of the window.
        window_width (float): The total width of the window.

        Returns:
        np.ndarray: The windowed and normalized pixel data.
        """
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        img = np.clip(pixel_array, lower, upper)
        return (img - lower) / (upper - lower)

    @staticmethod
    def load_raw_array(dicom_path):
        """
        Load a single DICOM file and return a normalized array.

        Parameters:
        dicom_path (str): Path to the DICOM file.

        Returns:
        array (np.ndarray): A normalized array of shape (H, W) ready for
            input to a neural network.
        """
        if isinstance(dicom_path, (str, Path)):
            dcm = pydicom.dcmread(dicom_path)
        elif isinstance(dicom_path, (bytes, bytearray)):
            dcm = pydicom.dcmread(io.BytesIO(dicom_path))
        else:
            raise TypeError(f"Unsupported type for dicom_path: {type(dicom_path)}")

        # Raw pixel values
        raw = dcm.pixel_array.astype(np.float32)

        # Rescale
        slope     = float(getattr(dcm, 'RescaleSlope', 1.0))
        intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
        scaled = raw * slope + intercept

        # WindowCenter / WindowWidth via percentiles
        pmin, pmax = np.percentile(scaled, [0.5, 99.5])
        wc = (pmin + pmax) / 2.0
        ww = pmax - pmin
        lower, upper = wc - ww / 2.0, wc + ww / 2.0
        
        # Window + normalize
        win = np.clip(scaled, lower, upper)
        norm = (win - lower) / (upper - lower + 1e-8)
        norm = np.clip(norm, 0.0, 1.0)

        return norm

    def load(self, dicom_path):
        """
        Load a single DICOM file and return a normalized tensor.

        Parameters:
        dicom_path (str): Path to the DICOM file.

        Returns:
        tensor (torch.Tensor): A normalized tensor of shape (1, H, W) ready for
            input to a neural network.
        """
        if isinstance(dicom_path, (bytes, bytearray)):
            dcm = pydicom.dcmread(io.BytesIO(dicom_path))
        else:
            dcm = pydicom.dcmread(dicom_path)
        # Read window values (handle MultiValue)
        if 'WindowCenter' in dcm:
            wc = float(dcm.WindowCenter[0] if isinstance(dcm.WindowCenter, pydicom.multival.MultiValue) else dcm.WindowCenter)
        else:
            wc = self.default_center
        if 'WindowWidth' in dcm:
            ww = float(dcm.WindowWidth[0] if isinstance(dcm.WindowWidth, pydicom.multival.MultiValue) else dcm.WindowWidth)
        else:
            ww = self.default_width
        pixel_array = dcm.pixel_array.astype(np.float32)
        windowed = self.window_image(pixel_array, wc, ww)

        # expand channels
        """
        # Convert to 1 channel
        img = np.expand_dims(windowed, axis=0)  # (1, H, W)
        tensor = torch.from_numpy(img)

        # apply torchvision transforms
        return self.transform(tensor)
        """

        # Convert to 3 channels
        img_3c = np.stack([windowed, windowed, windowed], axis=-1)  # (H, W, 3)
        pil = Image.fromarray((img_3c * 255).astype(np.uint8))
        
        if pil.mode != "RGB":
            pil = pil.convert("RGB")

        # apply torchvision transforms
        tensor = self.transform(pil) 
        return tensor

    def __call__(self, dicom_path):
        return self.load(dicom_path)