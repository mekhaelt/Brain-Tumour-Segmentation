# Brain Tumour Segmentation

This project focuses on automated brain tumor segmentation using advanced deep learning and computer vision techniques applied to 3D medical imaging. Traditionally, this task is time-consuming and manually intensive for radiologists, often requiring expert annotation across multiple imaging modalities. By automating this process, the project aims to drastically reduce segmentation time while improving accuracy and consistency—enabling faster diagnoses, more effective treatment planning, and scalable analysis in both clinical and research settings.

I used a custom dataset comprising over 5,000 3D MRI scans from both pre-treatment and post-treatment glioma patients. These scans include T1, T1ce, T2, and FLAIR modalities, and are preprocessed to ensure alignment, normalization, and compatibility with various segmentation frameworks.

## Installation

1. Clone the Repository:
```bash
   git clone https://github.com/mekhaelt/Brain-Tumour-Segmentation
   cd Brain-Tumour-Segmentation
```
2. Set up the Backend:
- Create and activate a virtual environment
```bash
  python -m venv venv
  venv\Scripts\activate
```
- Install required Python packages
```bash
  pip install -r requirements.txt
```
3. Train Model
When training the MedSAM model use the following command:
```
python train.py -net sam -mod sam_lora -exp_name ... -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -b 1 -dataset brats -thd True  -data_path data -w 8 -four_chan True 
```
NOTE: After running the training command, 'sam_vit_b_01ec64.pth' will be downloaded. If pretrained weights are not downloaded propperly, you cand do it manually through this link and store it in 'checkpoint/sam/'.

When training the nnUNet model first extract the datasets fingerprint with:
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
Then train the model with:
```
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD
```
When training the SwinUNETR model use :
```
train.py
```

## Model Overview

### 🟣 MedSAM
MedSAM is a medical adaptation of Meta AI’s Segment Anything Model, designed to handle a wide range of segmentation tasks with minimal fine-tuning. In this project, it is extended to 3D medical imaging by training on volumetric MRI data using point-based prompts as guidance signals. The architecture builds on a Vision Transformer encoder that generates patch-wise embeddings, followed by a lightweight mask decoder that predicts segmentation masks conditioned on spatial prompts. In the 3D adaptation, slices are processed either independently or through pseudo-3D strategies, enabling the model to generalize across modalities with minimal supervision. Its promptable nature enables flexible interaction with minimal input, making it ideal for clinical scenarios where annotation is sparse or expensive.

### 🟩 nnUNet
nnUNet, or no new net, is a self-adapting framework built upon the classic U-Net architecture. The core model consists of an encoder-decoder structure with symmetric skip connections, where the encoder down-samples feature maps through convolution and pooling layers, and the decoder reconstructs the segmentation map through transposed convolutions and concatenation with encoder features. What sets nnUNet apart is its automated pipeline: it dynamically configures preprocessing (such as intensity normalization and resampling), chooses between 2D, 3D, or cascade architectures based on dataset size and resolution, and optimizes training hyperparameters without manual tuning. This makes nnUNet exceptionally strong as a baseline for medical image segmentation, offering near state-of-the-art performance on a wide variety of datasets with little intervention.

### 🟦 SwinUNETR
SwinUNETR integrates the hierarchical structure of the Swin Transformer with a U-Net inspired encoder-decoder layout. The model first divides the 3D volume into non-overlapping patches and processes them using a series of Swin Transformer blocks. These blocks use window-based multi-head self-attention and shifted windowing to capture both local and global context efficiently while maintaining linear computational complexity with respect to input size. The encoder outputs are passed to a convolutional decoder via skip connections at multiple resolutions, enabling the model to reconstruct high-fidelity segmentation maps. By combining the representational power of transformers with the localization strength of U-Net-like structures, SwinUNETR is particularly effective at modeling complex tumor structures and boundaries in high-resolution MRI volumes.

## Dataset

The dataset used consists of over 5,000 3D MRI volumes of glioma patients both pre- and post-treatment. It includes:
- **Modalities**: T1, T1ce, T2, FLAIR
- **Annotations**: Expert-labeled tumor segmentations
- **Format**: NIfTI (.nii.gz) volumes
- **Preprocessing**: Resampling, intensity normalization, co-registration

> Note: The dataset is not publicly available due to privacy constraints.


To evaluate the effectiveness of modern segmentation architectures, I implemented and compared three state-of-the-art models:


| Model     | Mean Dice Score |
| --------- | --------------- |
| MedSAM    | 0.718           |
| nnUNet    | 0.806           |
| SwinUNETR | 0.825           |

