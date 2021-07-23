import json
import sys

import albumentations as alb
import cv2
import numpy as np

sys.path.append('../')
import aws_helper


with open('./categories.json', 'r') as f:
    categories_of_interest = json.load(f)


def resize_and_pad_img(img: np.ndarray, img_size: int = 224):
    """
    :param img: image as np array (of either int or floats)
    :param img_size: pixels in output side length
    :return: image resized to img_size and converted to a square by adding padding
    """
    resize_and_pad_transform = alb.Compose(
        [
            alb.LongestMaxSize(
                max_size=img_size,
                interpolation=cv2.INTER_CUBIC,
            ),
            alb.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
            ),
        ]
    )
    return resize_and_pad_transform(image=img).get('image')


def normalize_img(img: np.ndarray):
    """
    :param img: input image as np array
    :return: normalized image (convert from 0-255 RGB values to floats, then normalize by
    subtracting the mean and dividing by std deviation
    """
    return alb.Normalize(always_apply=True)(image=img).get('image')

def main():
