# https://medium.com/analytics-vidhya/pytorch-implementation-of-semantic-segmentation-for-single-class-from-scratch-81f96643c98c
# https://www.learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
# https://www.kaggle.com/dhananjay3/image-segmentation-from-scratch-in-pytorch
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import find_mask_contour_area
from PIL import Image
import torch
import torchvision.transforms as T

base_path = Path(__file__).parent.parent


abs_image_dir = Path(base_path / "data/lettuce.jpeg").resolve()
print(abs_image_dir)

img = cv2.imread(str(abs_image_dir))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
