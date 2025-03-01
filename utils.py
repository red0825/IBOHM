import random
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
from scipy.ndimage import zoom

# Set random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 3D NIfTI Dataset Class
# This class loads NIfTI images and extracts 2D slices for training.
# Domain distinction is based on the file name: 'eid' indicates UKB data, while others belong to ACDC.
class NiftiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []

        for file_name in os.listdir(root_dir):
            if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                label = 0 if file_name.startswith('eid') else 1  # UKB = 0, ACDC = 1
                nii_image = nib.load(os.path.join(root_dir, file_name))
                H, W, D = nii_image.get_fdata().shape
                for i in range(D):
                    self.file_paths.append((os.path.join(root_dir, file_name), i, label))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, slice_info, label = self.file_paths[idx]
        image = self._load_and_process_image(file_path, slice_info)
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension -> shape: (1, 256, 256)

        if self.transform:
            image = self.transform(image)

        image = self.normalize_image(image)  # Normalize intensity values to [0, 1]
        return image, label  # Return preprocessed image and domain label (UKB: 0, ACDC: 1)
    
    def _load_and_process_image(self, file_path, slice_info):
        """Load and process a single slice from a NIfTI image.
        
        - Extracts a 2D slice from a 3D volume.
        - Resizes the image to (256, 256) using bilinear interpolation (order=2).
        """
        nii_image = nib.load(file_path)
        image = nii_image.get_fdata()[:, :, slice_info]  # Extract the selected slice
        
        # Compute zoom factors for resizing to 256x256
        zoom_factors = (256 / image.shape[0], 256 / image.shape[1])
        return np.asarray(zoom(image, zoom_factors, order=2), dtype=np.float32)

    def normalize_image(self, image):
        """Normalize the image to the range [0, 1] to stabilize training.
        
        - Prevents division by zero using eps (1e-8).
        - Ensures consistent intensity scaling across different domains.
        """
        image = image.squeeze().numpy()
        eps = 1e-8  # Avoid division by zero
        return torch.from_numpy((image - np.min(image)) / (np.max(image) - np.min(image) + eps)).unsqueeze(0)