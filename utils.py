import random
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import torch.nn.functional as F
from skimage.exposure import match_histograms
from scipy.ndimage import zoom, gaussian_filter

# Set random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# Apply Gaussian blur to an image
def apply_blur(image, sigma=0.3):
    return gaussian_filter(image, sigma=sigma)

# Perform histogram matching between source and target images
def histogram_matching(source_img, target_img):
    try:
        return match_histograms(source_img, target_img)
    except Exception as e:
        print(f"Histogram matching failed: {e}")
        return source_img

# Load NIfTI file and extract image data, affine, and header
def load_nifti(file_path):
    try:
        img = nib.load(file_path)
        return img.get_fdata(), img.affine, img.header
    except Exception as e:
        print(f"Error loading NIfTI file {file_path}: {e}")
        raise e

# Normalize image intensity to range [0,1]
def normalize_image(image):
    eps = 1e-8
    return (image - np.min(image)) / (np.max(image) - np.min(image) + eps)

# Resize image to target shape using linear interpolation
def resize_image(image, target_shape):
    factors = [t / s for s, t in zip(image.shape, target_shape)]
    return zoom(image, factors, order=2)

# Classify each 2D slice of a 3D image using the provided model
def classify_3d_image(image, model, device):
    model.eval()
    D = image.shape[2]
    predictions = []
    with torch.no_grad():
        for d in range(D):
            slice_image = normalize_image(resize_image(image[:, :, d], (256, 256)))
            slice_tensor = torch.tensor(slice_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            _, output = model(slice_tensor)
            prob = F.softmax(output, dim=1).cpu().numpy()[0, 0]
            predictions.append(prob)
    return np.mean(predictions)