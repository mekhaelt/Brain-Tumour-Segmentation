import os
import glob
import shutil
import json

# === Paths ===
raw_data_root = "raw_data/MICCAI-LH-BraTS2025-MET-Challenge-Training"
output_root = "nnunet/data/nnUNet_raw_data_base/nnUNet_raw_data/Task501_BrainTumor"
imagesTr_dir = os.path.join(output_root, "imagesTr")
labelsTr_dir = os.path.join(output_root, "labelsTr")
output_json_path = os.path.join(output_root, "dataset.json")

os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)

# === Define modality/channel mapping ===
modality_to_channel = {
    "t2f": "0000",  # FLAIR
    "t1n": "0001",  # T1
    "t1c": "0002",  # T1ce
    "t2w": "0003",  # T2
}

# === Label map for dataset.json ===
labels_dict = {
    "0": "background",
    "1": "tumor_core",
    "2": "edema",
    "4": "enhancing_tumor"
}

# === Track valid training cases ===
training_entries = []

# === Main data copy loop ===
for case_folder in glob.glob(os.path.join(raw_data_root, "BraTS-MET-*")):
    case_id_full = os.path.basename(case_folder)  # BraTS-MET-00577-000
    parts = case_id_full.split("-")
    
    if len(parts) < 4 or parts[3] != "000":
        continue  # skip if not the -000 version

    patient_id = parts[2]
    case_id = f"BraTS-MET-{patient_id}"  # Remove "-000"

    # Copy images
    valid = True
    for modality, channel in modality_to_channel.items():
        src = os.path.join(case_folder, f"{case_id_full}-{modality}.nii.gz")
        dst = os.path.join(imagesTr_dir, f"{case_id}_{channel}.nii.gz")
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"[⚠️] Missing modality {modality} for {case_id_full}, skipping case.")
            valid = False
            break

    # Copy segmentation label
    seg_src = os.path.join(case_folder, f"{case_id_full}-seg.nii.gz")
    seg_dst = os.path.join(labelsTr_dir, f"{case_id}.nii.gz")
    if not os.path.exists(seg_src):
        print(f"[⚠️] Missing segmentation for {case_id_full}, skipping case.")
        valid = False
    else:
        shutil.copy(seg_src, seg_dst)

    # Add to dataset if valid
    if valid:
        training_entries.append({
            "image": f"./imagesTr/{case_id}",
            "label": f"./labelsTr/{case_id}.nii.gz"
        })

# === Create dataset.json ===
dataset_dict = {
    "name": "BrainTumorSegmentation",
    "description": "nnU-Net task for segmenting brain tumors from BraTS 2025 MRI data",
    "tensorImageSize": "3D",
    "modality": {
        "0": "FLAIR",
        "1": "T1",
        "2": "T1ce",
        "3": "T2"
    },
    "labels": labels_dict,
    "numTraining": len(training_entries),
    "file_ending": ".nii.gz",
    "training": training_entries
}

with open(output_json_path, "w") as f:
    json.dump(dataset_dict, f, indent=4)

print("✅ All files processed and dataset.json generated.")
