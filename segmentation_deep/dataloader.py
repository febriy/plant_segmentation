# visualization library
import cv2
from matplotlib import pyplot as plt

# data storing library
import numpy as np
import pandas as pd

# torch libraries
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler

# architecture and data split library
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

# augmenation library
from albumentations import (
    HorizontalFlip,
    ShiftScaleRotate,
    Normalize,
    Resize,
    Compose,
    GaussNoise,
)
from albumentations.pytorch import ToTensor

# others
import os
import pdb
import time
import warnings
import random
from tqdm import tqdm_notebook as tqdm
import concurrent.futures
from pathlib import Path
import PIL

# warning print supression
warnings.filterwarnings("ignore")

# *****************to reproduce same results fixing the seed and hash*******************
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


base_path = Path(__file__).parent.parent
data_path = Path(base_path / "data/").resolve()

df = pd.read_csv(data_path / "Metadata.csv")

# location of original and mask image
img_fol = data_path / "train-128"
mask_fol = data_path / "train_masks_bw-128"

# imagenet mean/std will be used as the resnet backbone is trained on imagenet stats
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# during traning/val phase make a list of transforms to be used.
# input-->"phase",mean,std
# output-->list
def get_transform(phase, mean, std):
    list_trans = []
    if phase == "train":
        list_trans.extend([HorizontalFlip(p=0.5)])
    list_trans.extend(
        [Normalize(mean=mean, std=std, p=1), ToTensor()]
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


"""divide data into train and val and return the dataloader depending upon train or val phase."""


def PlantDataloader(df, img_fol, mask_fol, mean, std, phase, batch_size, num_workers):
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=69)
    df = df_train if phase == "train" else df_valid
    for_loader = PlantDataset(df, img_fol, mask_fol, mean, std, phase)
    dataloader = DataLoader(
        for_loader, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    return dataloader
