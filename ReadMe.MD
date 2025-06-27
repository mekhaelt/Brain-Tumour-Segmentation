# Brain Tumour Segmentation

This project focuses on automated brain tumor segmentation using advanced deep learning and computer vision techniques applied to 3D medical imaging. Traditionally, this task is time-consuming and manually intensive for radiologists, often requiring expert annotation across multiple imaging modalities. By automating this process, the project aims to drastically reduce segmentation time while improving accuracy and consistency, enabling faster diagnoses, more effective treatment planning, and scalable analysis in both clinical and research settings.

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
3. Training
To train the MedSAM model run:
```
python train.py -net sam -mod sam_lora -exp_name ... -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -b 1 -dataset brats -thd True  -data_path data -w 8 -four_chan True 
```
NOTE: After running the training command, 'sam_vit_b_01ec64.pth' will be downloaded. If pretrained weights are not downloaded properly, you can do it manually [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and store it in 'checkpoint/sam/'.

To train the nnUNet model first extract the datasets fingerprint with:
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
Then run:
```
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD
```
When training the SwinUNETR model run :
```
train.py
```

## Model Overview

### 🟣 MedSAM
MedSAM is a medical adaptation of Meta AI’s Segment Anything Model, extended to 3D brain MRI segmentation. It uses a Vision Transformer encoder to extract spatial features and a lightweight decoder to generate masks based on point prompts. In this project, slices are processed in pseudo-3D to handle volumetric data. Its promptable design enables efficient segmentation with minimal input, making it well-suited for interactive or sparse-annotation scenarios.

### 🟩 nnUNet
nnUNet is a self-configuring segmentation framework built on a U-Net backbone. It features a symmetric encoder-decoder with skip connections and automatically adjusts preprocessing, architecture, and training settings based on dataset properties. It supports 2D, 3D, and cascade variants, and consistently delivers strong performance without manual tuning, making it a reliable benchmark in medical image segmentation.

### 🟦 SwinUNETR
SwinUNETR combines Swin Transformers with a U-Net-style encoder-decoder. It splits the input volume into patches, applies window-based self-attention for efficient context modeling, and uses skip connections to reconstruct fine-grained segmentations. This hybrid approach captures both local detail and global structure, making it especially effective for complex 3D tasks like tumor segmentation in MRI scans.

## Dataset

The dataset used consists of over 5,000 3D MRI volumes of glioma patients both pre- and post-treatment. It includes:
- **Modalities**: T1, T1ce, T2, FLAIR
- **Annotations**: Expert-labeled tumor segmentations
- **Format**: NIfTI (.nii.gz) volumes
- **Preprocessing**: Resampling, intensity normalization, co-registration

> Note: The dataset is not publicly available due to privacy constraints.


To evaluate the effectiveness of modern segmentation architectures, I implemented and compared three state-of-the-art models:

## Results

| Model     | Mean Dice Score |
| --------- | --------------- |
| MedSAM    | 0.718           |
| nnUNet    | 0.806           |
| SwinUNETR | 0.825           |

SwinUNETR achieved the highest mean Dice score, demonstrating superior performance in segmenting brain tumors from 3D MRI scans. nnUNet also performed well, validating its reputation as a strong baseline for medical image segmentation. While MedSAM scored slightly lower, its flexibility and prompt-based interaction make it a promising tool for rapid or low-supervision scenarios. These results highlight the potential of modern deep learning models, especially transformer-based architectures, to enhance accuracy and efficiency in clinical imaging workflows. With further fine-tuning and data augmentation, all models show potential for real-world deployment in radiology and neuro-oncology.

## Future Plans

- Evaluate generalization performance on other conditions such as **intracranial meningioma** and **brain metastases**
- Experiment with additional transformer-based architectures, including the [3D TransUNet model](https://arxiv.org/pdf/2310.07781)
- Integrate uncertainty quantification or ensemble models for more robust clinical predictions

## References

- [Isensee et al., 2021] nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*.
- [Tang et al., 2023] MedSAM: Segment Anything for Medical Images. arXiv:2306.05396
- [Hatamizadeh et al., 2022] Swin UNETR: Transformers for semantic segmentation of brain tumors in MRI images. MICCAI 2022.







