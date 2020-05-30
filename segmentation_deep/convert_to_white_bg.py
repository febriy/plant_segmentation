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

img = cv2.imread(img_name_path, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_name_path, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)

img[thresh == 0] = 255

# print(mask)
# coloured = img.copy()
# coloured[mask == 255] = (255, 255, 255)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
fig.suptitle("predicted_mask//original_mask")


ax1.imshow(img)
ax2.imshow(thresh)

plt.show()
