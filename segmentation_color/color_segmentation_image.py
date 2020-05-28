from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import find_mask_contour_area

base_path = Path(__file__).parent.parent


def open_image(image_dir):
    abs_image_dir = Path(base_path / image_dir).resolve()

    plant = cv2.imread(str(abs_image_dir))
    plant = cv2.cvtColor(plant, cv2.COLOR_BGR2RGB)
    hsv_plant = cv2.cvtColor(plant, cv2.COLOR_RGB2HSV)
    return hsv_plant, plant


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    data_path = Path(base_path / "data/").resolve()

    img_fol = data_path / "mydata-128"

    # Set threshold
    lower_green = (10, 10, 10)
    upper_green = (130, 255, 255)

    for idx, image_path in enumerate(list(img_fol.iterdir())):
        hsv_plant, plant = open_image(image_path)

        mask = cv2.inRange(hsv_plant, lower_green, upper_green)
        result = cv2.bitwise_and(plant, plant, mask=mask)

        # use contours to find mask area
        # https://stackoverflow.com/questions/57282935/how-to-detect-area-of-pixels-with-the-same-color-using-opencv
        area = find_mask_contour_area(plant, mask)
        print(area)

        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.show()
