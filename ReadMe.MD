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

python train.py -net sam -mod sam_lora -exp_name ... -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -b 1 -dataset brats -thd True  -data_path data -w 8 -four_chan True 

NOTE: After running the training command, 'sam_vit_b_01ec64.pth' will be downloaded. If pretrained weights are not downloaded propperly, you cand do it manually through this link and store it in 'checkpoint/sam/'.

When training the nnUNet model first extract the datasets fingerprint with:

nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

Then train the model with:

nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD

When trainnig the SwinUNETR model use train.py




To evaluate the effectiveness of modern segmentation architectures, I implemented and compared three state-of-the-art models:


| Model     | Mean Dice Score |
| --------- | --------------- |
| MedSAM    | 0.718           |
| nnUNet    | 0.806           |
| SwinUNETR | 0.825           |

