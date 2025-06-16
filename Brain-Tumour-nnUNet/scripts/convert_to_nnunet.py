from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import join, subdirs, subfiles, maybe_mkdir_p
from nnunetv2.paths import nnUNet_raw
from pathlib import Path

if __name__ == '__main__':
    """
    this dataset does not copy the data into nnunet format and just links to existing data. The dataset can only be 
    used from one machine because the paths in the dataset.json are hard coded
    """
    extracted_BraTS2024_GLI_dir = 'C:/Users/mekha/Desktop/Brain-Tumour-Detection/raw_data/training_data1_v2'
    nnunet_dataset_name = 'BraTS2024-BraTS-GLI'
    nnunet_dataset_id = 226
    dataset_name = f'Dataset{nnunet_dataset_id:03d}_{nnunet_dataset_name}'
    dataset_dir = (Path(nnUNet_raw) / dataset_name).as_posix()
    print('dataset_name: ', dataset_name)
    print('nnUNet_raw: ', nnUNet_raw )
    print('directory', dataset_dir)
    maybe_mkdir_p(dataset_dir)

    dataset = {}
    casenames = subdirs(extracted_BraTS2024_GLI_dir, join=False)
    for c in casenames:
        dataset[c] = {
            'label': join(extracted_BraTS2024_GLI_dir, c, c + '-seg.nii.gz').replace("\\", "/"),
            'images': [
                join(extracted_BraTS2024_GLI_dir, c, c + '-t1c.nii.gz').replace("\\", "/"),
                join(extracted_BraTS2024_GLI_dir, c, c + '-t1n.nii.gz').replace("\\", "/"),
                join(extracted_BraTS2024_GLI_dir, c, c + '-t2f.nii.gz').replace("\\", "/"),
                join(extracted_BraTS2024_GLI_dir, c, c + '-t2w.nii.gz').replace("\\", "/")
            ]
        }
    labels = {
        'background': 0,
        'SNFH': 1,
        'NETC': 2,
        'RC': 3,
        'ET': 4,
        
    }

    generate_dataset_json(
        dataset_dir,
        {
            0: 'T1C',
            1: "T1",
            2: "T2F",
            3: "T2W"
        },
        labels,
        num_training_cases=len(dataset),
        file_ending='.nii.gz',
        regions_class_order=None,
        dataset_name=dataset_name,
        reference='https://www.synapse.org/Synapse:syn53708249/wiki/627500',
        license='see https://www.synapse.org/Synapse:syn53708249/wiki/627508',
        dataset=dataset,
        description='This dataset does not copy the data into nnunet format and just links to existing data. '
                    'The dataset can only be used from one machine because the paths in the dataset.json are hard coded'
    )