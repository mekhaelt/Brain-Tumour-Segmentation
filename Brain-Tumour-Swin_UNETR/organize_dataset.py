import os
import json

def generate_brats_json(data_dir, output_path, max_samples=100):
    dataset = {"training": []}
    count = 0  

    for case in sorted(os.listdir(data_dir)):
        case_path = os.path.join(data_dir, case)
        if not os.path.isdir(case_path) or not case.startswith("BraTS"):
            continue

        entry = {
            "fold": 0,
            "image": [
                f"{case}/{case}-t2f.nii.gz",
                f"{case}/{case}-t1c.nii.gz",
                f"{case}/{case}-t1n.nii.gz",
                f"{case}/{case}-t2w.nii.gz",
            ],
            "label": f"{case}/{case}-seg.nii.gz"
        }

        dataset["training"].append(entry)
        count += 1
        if count >= max_samples:
            break

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Saved dataset JSON to: {output_path} ({count} samples)")

if __name__ == "__main__":
    input_dir = "C:/Users/mekha/Desktop/Brain-Tumour-Detection/raw_data/training_data1_v2"
    output_file = "C:/Users/mekha/Desktop/Brain-Tumour-Detection/Brain-Tumour-Swin_UNETR/dataset.json"

    generate_brats_json(input_dir, output_file)
