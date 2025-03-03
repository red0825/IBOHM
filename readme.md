# IBOHM: Domain Adaptation via Image-Based Optimization and Histogram Matching

## Overview

This project aims to adapt medical imaging data from a source domain to a target domain while preserving anatomical structures. The process leverages a **ResNet-based classifier** to assess domain distinction and an **Image-Based Optimization and Histogram Matching (IBOHM) generator** to transform images accordingly.

## Dataset

- The dataset consists of **3D NIfTI images** from two sources:
  - **UKB** images are labeled as `0`
  - **ACDC** images are labeled as `1`
- Each 3D image consists of multiple **2D slices**, which are used as input for training.
- Images differ in **resolution, contrast, and imaging protocols**, necessitating domain adaptation.
## Data Preprocessing

1. **Extracting 2D Slices**: Each 3D image is sliced along the last axis (depth).
2. **Resizing**: All slices are resized to **256×256** using bilinear interpolation.
3. **Normalization**: Pixel values are normalized to the range **[0, 1]**.

## Model Architecture (Classifier)

- The classifier is based on **ResNet-50**.
- The `forward` function returns **both feature maps** and the **final classification output**.
- The classification head consists of a fully connected layer.

## Training (Classifier)

- **Loss Function**: Cross-entropy loss.
- **Optimizer**: Adam.
- **Early Stopping**: Training stops if validation loss does not improve for 100 epochs.

## Evaluation (Classifier)

- The trained model is evaluated on a separate test set.
- Metrics:
  - **Accuracy**: Measures correct classification.
  - **Loss**: Average cross-entropy loss on the test set.

## Generator: Domain Adaptation Process

The domain adaptation workflow consists of the following steps:

1. **Data Loading & Preprocessing**
   - NIfTI images are loaded, and individual 2D slices are extracted.
   - Intensity normalization is applied to scale pixel values to the range [0,1].
   - **Histogram matching** is used to align intensity distributions between source and target images.
   - **Gaussian blur** is optionally applied to smooth image noise.

2. **Feature-Based Image Transformation**
   - A **ResNet-50 classifier** extracts high-level features from both source and target domain images.
   - An optimization process is performed to **minimize feature differences** while preserving essential image structures.
   - Early stopping is used to prevent overfitting during optimization.

3. **Classification & Validation**
   - The transformed image is classified using the **ResNet-50 classifier** to evaluate domain similarity.
   - If the **classification confidence (softmax probability)** exceeds a predefined **threshold (0.5)**, the transformed image is saved.
   - If not, another iteration of transformation is performed using a different target reference image.

4. **Saving & Output**
   - Successfully transformed images are saved in the specified output directory as **NIfTI files**.
   - The model ensures that the transformed images retain **clinical relevance** while aligning with the target domain.

## Usage

### 1. Set Up Environment

- **GPU**: NVIDIA GeForce 2080 Ti
- **CUDA Version**: 10.1
- **Python Version**: 3.8.2

```bash
pip install -r requirements.txt
```

### 2. Run Model Training (Classifier)

```bash
python train_classifier.py --data_path /path/to/train_data --model_save_path /path/to/save_model
```

### 3. Run Model Testing (Classifier)

```bash
python test_classifier.py --data_path /path/to/test_data --model_path /path/to/saved_model/best_model.pth
```

### 4. Run Generator

```bash
python IBOHM_generator.py --source_folder path/to/source --target_folder path/to/target --output_folder path/to/output --model_folder path/to/model
```

## File Structure

```
project_root/
│── IBOHM_generator.py      # Generator part
│── train_classifier.py     # Training part
│── test_classifier.py      # Testing part
│── resnet.py               # ResNet model definition
│── utils.py                # Utility functions
│── README.md               # Project documentation
```

## Acknowledgments
This project is inspired by or based on the following works:

- **Original Paper**: Parreño et al., *Deidentifying MRI Data Domain by Iterative Backpropagation*, STACOM 2020, 2021.
- Codebase Reference: [Here](https://github.com/MarioProjects/MnMsCardiac)