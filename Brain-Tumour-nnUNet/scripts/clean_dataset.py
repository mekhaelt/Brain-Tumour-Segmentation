import os

imagesTr = "nnunet/data/nnUNet_raw_data/Task501_BrainTumor/imagesTr"
case_ids = [f.split('_')[0] for f in os.listdir(imagesTr) if f.endswith(".nii.gz")]
duplicates = set([x for x in case_ids if case_ids.count(x) > 4])  # More than 4 files = possibly repeated

print("Potential duplicate base names:", duplicates)

