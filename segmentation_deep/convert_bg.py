import os
from pathlib import Path
import pandas as pd
import cv2
from torch.utils.data import DataLoader, Dataset, sampler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

base_path = Path(__file__).parent.parent
data_path = Path(base_path / "data/").resolve()

img_name_path = str(data_path / "train-256/ara2013_plant001_rgb.png")
mask_name_path = str(data_path / "train_masks-256/ara2013_plant001_label.png")
bg_name_path = str(data_path / "mydata-256/picam.png")

img = cv2.imread(img_name_path, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_name_path, cv2.COLOR_BGR2RGB)  # , cv2.COLOR_BGR2GRAY)
bg = cv2.imread(bg_name_path, cv2.COLOR_BGR2RGB)

## Change to white
# ret, thresh = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
# img[thresh == 0] = 255


# change to another background
img[mask == 0] = bg[mask == 0]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
fig.suptitle("predicted_mask//original_mask")


ax1.imshow(mask)
ax2.imshow(img)

plt.show()
