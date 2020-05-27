from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb
from PIL import Image
import os

base_path = Path(__file__).parent.parent
image_dir = Path(base_path / "data/lettuce.jpeg").resolve()

plant = cv2.imread(str(image_dir))
plant = cv2.cvtColor(plant, cv2.COLOR_BGR2RGB)
hsv_plant = cv2.cvtColor(plant, cv2.COLOR_RGB2HSV)
plt.imshow(plant)
plt.show()

# Set threshold
lower_green = (25, 50, 25)
upper_green = (120, 255, 255)


def view_pixel_color_graph(image):
    pixel_colors = image.reshape((np.shape(image)[0] * np.shape(image)[1], 3))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    h, s, v = cv2.split(image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.scatter(
        h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker="."
    )
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()


def show_colour_space(lower_color, upper_color):
    lo_square = np.full((10, 10, 3), lower_green, dtype=np.uint8) / 255.0
    do_square = np.full((10, 10, 3), upper_green, dtype=np.uint8) / 255.0

    plt.subplot(1, 2, 1)
    plt.imshow(hsv_to_rgb(do_square))
    plt.subplot(1, 2, 2)
    plt.imshow(hsv_to_rgb(lo_square))
    plt.show()


if __name__ == "__main__":
    mask = cv2.inRange(hsv_plant, lower_green, upper_green)
    result = cv2.bitwise_and(plant, plant, mask=mask)

    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()
