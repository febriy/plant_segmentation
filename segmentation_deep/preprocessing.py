import os
from pathlib import Path
from matplotlib import pyplot as plt
import PIL
import concurrent.futures

base_path = Path(__file__).parent.parent
data_path = Path(base_path / "data/").resolve()


def resize_mask(fn):
    image_file = PIL.Image.open(fn).resize((128, 128))
    image_file.save((fn.parent.parent) / "train_masks-128" / fn.name)


def recolor_mask(fn):
    img = PIL.Image.open(fn)
    thresh = 5
    fn_thresh = lambda x: 255 if x > thresh else 0
    img = img.convert("L").point(fn_thresh, mode="1")

    img.save((fn.parent.parent) / "train_masks_bw-128" / fn.name)


def resize_img(fn):
    PIL.Image.open(fn).resize((128, 128)).save(
        (fn.parent.parent) / "train-128" / fn.name
    )


def resize_myimg(fn):
    PIL.Image.open(fn).resize((128, 128)).save(
        (fn.parent.parent) / "mydata-128" / fn.name
    )


if __name__ == "__main__":
    # Create new folders for edited images
    (data_path / "train_masks-128").mkdir(exist_ok=True)
    (data_path / "train_masks_bw-128").mkdir(exist_ok=True)
    (data_path / "train-128").mkdir(exist_ok=True)
    (data_path / "mydata-128").mkdir(exist_ok=True)

    files = list((data_path / "train_masks_png").iterdir())
    with concurrent.futures.ThreadPoolExecutor(8) as e:
        e.map(resize_mask, files)

    files = list((data_path / "train_masks-128").iterdir())
    with concurrent.futures.ThreadPoolExecutor(8) as e:
        e.map(recolor_mask, files)

    files = list((data_path / "train").iterdir())
    with concurrent.futures.ThreadPoolExecutor(8) as e:
        e.map(resize_img, files)

    files = list((data_path / "mydata_png").iterdir())
    with concurrent.futures.ThreadPoolExecutor(8) as e:
        e.map(resize_myimg, files)
