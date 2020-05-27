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


# Set threshold
lower_green = (25, 50, 25)
upper_green = (120, 255, 255)


if __name__ == "__main__":
    hsv_plant, plant = open_image("data/lettuce.jpeg")
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
