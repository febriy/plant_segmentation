from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

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


def find_mask_contour_area(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv2.imshow("opening", opening)
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    area = 0
    for c in cnts:
        area += cv2.contourArea(c)
        cv2.drawContours(plant, [c], 0, (0, 0, 0), 2)

    return area


if __name__ == "__main__":
    hsv_plant, plant = open_image("data/lettuce.jpeg")
    mask = cv2.inRange(hsv_plant, lower_green, upper_green)
    result = cv2.bitwise_and(plant, plant, mask=mask)

    # use contours to find mask area
    # https://stackoverflow.com/questions/57282935/how-to-detect-area-of-pixels-with-the-same-color-using-opencv
    area = find_mask_contour_area(mask)
    print(area)

    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()
