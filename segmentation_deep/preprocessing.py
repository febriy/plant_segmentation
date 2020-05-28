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

# we convert the high resolution image mask to 128*128 for starting for the masks.
(data_path / "train_masks-128").mkdir(exist_ok=True)
(data_path / "train_masks_bw-128").mkdir(exist_ok=True)


def resize_mask(fn):
    image_file = PIL.Image.open(fn).resize((128, 128))
    image_file.save((fn.parent.parent) / "train_masks-128" / fn.name)


def recolor_mask(fn):
    img = PIL.Image.open(fn)
    thresh = 5
    fn_thresh = lambda x: 255 if x > thresh else 0
    img = img.convert("L").point(fn_thresh, mode="1")

    img.save((fn.parent.parent) / "train_masks_bw-128" / fn.name)


files = list((data_path / "train_masks_png").iterdir())
with concurrent.futures.ThreadPoolExecutor(8) as e:
    e.map(resize_mask, files)

files = list((data_path / "train_masks-128").iterdir())
with concurrent.futures.ThreadPoolExecutor(8) as e:
    e.map(recolor_mask, files)

# # # we convert the high resolution input image to 128*128
(data_path / "train-128").mkdir(exist_ok=True)


def resize_img(fn):
    PIL.Image.open(fn).resize((128, 128)).save(
        (fn.parent.parent) / "train-128" / fn.name
    )


files = list((data_path / "train").iterdir())
with concurrent.futures.ThreadPoolExecutor(8) as e:
    e.map(resize_img, files)
