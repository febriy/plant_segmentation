import os
from pathlib import Path
import pandas as pd
import cv2
from torch.utils.data import DataLoader, Dataset, sampler
from sklearn.model_selection import train_test_split

# image augmantation library
from albumentations import (
    HorizontalFlip,
    Normalize,
    Compose,
    HueSaturationValue,
    RandomContrast,
    RandomBrightness,
    RGBShift,
    CenterCrop,
    RandomGamma,
)
from albumentations.pytorch import ToTensor

# imagenet mean/std will be used as the resnet backbone is trained on imagenet stats
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_transform(phase: str, mean: float, std: float) -> list:
    """Make a list of transformations to be used."""
    list_trans = []
    if phase == "train":
        list_trans.extend(
            [
                HorizontalFlip(p=0.5),
                RGBShift(r_shift_limit=10, g_shift_limit=40, b_shift_limit=10, p=0.5),
                HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=150, val_shift_limit=10, p=0.5
                ),
                RandomContrast(limit=1.1, p=0.5),
                RandomBrightness(limit=1.1, p=0.5),
                CenterCrop(height=64, width=64, p=0.5),
            ]
        )

    list_trans.extend(
        [Normalize(mean=mean, std=std, p=1), ToTensor(),]
    )  # normalizing the data & then converting to tensors
    list_trans = Compose(list_trans)
    return list_trans


class PlantDataset(Dataset):
    def __init__(self, df, img_fol, mask_fol, mean, std, phase):
        self.fname = df.iloc[:, 0].values.tolist()
        self.img_fol = img_fol
        self.mask_fol = mask_fol
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transform = get_transform(phase, mean, std)

    def __getitem__(self, idx):
        name = self.fname[idx]
        img_name_path = os.path.join(self.img_fol, name + "_rgb.png")
        mask_name_path = (
            img_name_path.split(".")[0]
            .replace("train-128", "train_masks_bw-128")
            .replace("_rgb", "_label.png")
        )

        img = cv2.imread(img_name_path)
        mask = cv2.imread(mask_name_path, cv2.IMREAD_GRAYSCALE)
        augmentation = self.transform(image=img, mask=mask)
        img_aug = augmentation["image"]  # [3,128,128] type:Tensor
        mask_aug = augmentation["mask"]  # [1,128,128] type:Tensor
        return img_aug, mask_aug

    def __len__(self):
        return len(self.fname)


def PlantDataloader(df, img_fol, mask_fol, mean, std, phase, batch_size, num_workers):
    """divide data into train and val and return the dataloader depending upon train or val phase."""
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=69)
    df = df_train if phase == "train" else df_valid
    for_loader = PlantDataset(df, img_fol, mask_fol, mean, std, phase)
    dataloader = DataLoader(
        for_loader, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    return dataloader


class PlantToInfer(Dataset):
    def __init__(self, img_fol, mean, std):
        self.img_fol = img_fol
        self.mean = mean
        self.std = std
        self.transform = get_transform("val", mean, std)

    def __getitem__(self, idx):
        img_name_path = str(list(self.img_fol.iterdir())[idx])
        print("img_name_path", img_name_path)

        img = cv2.imread(img_name_path)
        mask = cv2.imread(img_name_path, cv2.IMREAD_GRAYSCALE)
        augmentation = self.transform(image=img, mask=mask)
        img_aug = augmentation["image"]  # [3,128,128] type:Tensor
        mask_aug = augmentation["mask"]  # [1,128,128] type:Tensor
        return img_aug, mask_aug

    def __len__(self):
        return len(list(self.img_fol.iterdir()))


def PlantToInferloader(img_fol, mean, std, batch_size, num_workers):
    for_loader = PlantToInfer(img_fol, mean, std)
    dataloader = DataLoader(
        for_loader, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    return dataloader
