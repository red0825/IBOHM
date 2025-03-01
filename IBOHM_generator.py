import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from skimage.exposure import match_histograms
from scipy.ndimage import zoom, gaussian_filter
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from resnet import ResNet, Bottleneck

# Disable logging to avoid unnecessary output
logging.disable(logging.CRITICAL)

# Set random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Perform histogram matching between source and target images
def histogram_matching(source_img, target_img):
    try:
        return match_histograms(source_img, target_img)
    except Exception as e:
        logging.warning(f"Histogram matching failed: {e}")
        return source_img

# Load NIfTI file and extract image data, affine, and header
def load_nifti(file_path):
    try:
        img = nib.load(file_path)
        return img.get_fdata(), img.affine, img.header
    except Exception as e:
        logging.error(f"Error loading NIfTI file {file_path}: {e}")
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

# Image transformation class using feature-based optimization
class ImageBackward:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device).eval()
        self.device = device

    def apply(self, source_image, target_image):
        H, W, D = source_image.shape
        resized_target = resize_image(target_image, (256, 256, D))

        source_slices = [normalize_image(resize_image(source_image[:, :, d], (256, 256))) for d in range(D)]
        target_slices = [normalize_image(resized_target[:, :, d]) for d in range(D)]

        source_tensor = torch.tensor(np.stack(source_slices, axis=0), dtype=torch.float32).unsqueeze(1).to(self.device)
        target_tensor = torch.tensor(np.stack(target_slices, axis=0), dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            target_features, _ = self.model(target_tensor)

        transformed_image = source_tensor.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([transformed_image], lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 10

        for iteration in range(100):
            optimizer.zero_grad()
            transformed_features, _ = self.model(transformed_image)
            feature_loss = criterion(transformed_features[3], target_features[3])
            maintain_loss = F.l1_loss(transformed_image, source_tensor)
            loss = feature_loss + 0.5 * maintain_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter > early_stop_patience:
                break

        final_image = transformed_image.detach().cpu().squeeze().numpy()
        return np.stack([resize_image(final_image[d, :, :], (H, W)) for d in range(D)], axis=2)

# Apply Gaussian blur to an image
def apply_blur(image, sigma=0.3):
    return gaussian_filter(image, sigma=sigma)

# Process image pairs and save transformed images
def process_image_pairs(source_folder, target_folder, output_folder, model, device, threshold=0.5, max_attempts=2):
    source_files = sorted(Path(source_folder).glob('*.nii.gz'))
    target_files = sorted(Path(target_folder).glob('*.nii.gz'))

    if not source_files or not target_files:
        logging.error("No source or target files found.")
        return

    transformer = ImageBackward(model=model, device=device)

    for source_file in source_files:
        source_image, source_affine, source_header = load_nifti(source_file)
        transformed_image = source_image.copy()
        success = False
        attempts = 0

        while not success and attempts < max_attempts:
            target_file = random.choice(target_files)
            target_image, _, _ = load_nifti(target_file)

            transformed_image = histogram_matching(apply_blur(transformed_image), target_image)
            transformed_image = transformer.apply(transformed_image, target_image)
            avg_prob = classify_3d_image(transformed_image, model, device)

            if avg_prob > threshold:
                nib.save(nib.Nifti1Image(transformed_image, source_affine, source_header), str(Path(output_folder) / source_file.name))
                success = True

            attempts += 1

        if not success:
            nib.save(nib.Nifti1Image(transformed_image, source_affine, source_header), str(Path(output_folder) / source_file.name))
        torch.cuda.empty_cache()

# Main function to execute the pipeline
def main():
    """
    This function sets up the environment, loads the pre-trained classifier model,
    and processes source domain images by adapting their characteristics to match
    the target domain images.
    
    - The `source_folder` contains images from the source domain.
    - The `target_folder` contains images from the target domain, used for adaptation.
    - The `output_folder` stores the transformed images with adapted domain characteristics.
    - The classifier model, stored in `model_folder`, is used to guide the adaptation process.
    """
    set_random_seed(42)
    source_folder = 'source_domain_folder'
    target_folder = 'target_domain_folder'
    output_folder = 'output_folder'
    model_folder = 'model_folder'

    os.makedirs(output_folder, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained classifier model
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model_weights.pth'), map_location=device))
    model.to(device).eval()

    # Process source domain images with target domain characteristics
    process_image_pairs(source_folder, target_folder, output_folder, model, device)
