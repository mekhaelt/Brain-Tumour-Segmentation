import os
from monai import data
from monai.transforms import (
    Compose,
    LoadImaged,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from utils import datafold_read
import json

def get_loader(batch_size, data_dir, json_list, fold, roi):
    train_files, validation_files = datafold_read(datalist=json_list, basedir=data_dir, fold=fold)
    #didnt wrap json_list
    train_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            k_divisible=roi,
            allow_smaller=True,
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=roi, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])

    val_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    train_ds = data.Dataset(data=train_files, transform=train_transform)
    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, val_loader
