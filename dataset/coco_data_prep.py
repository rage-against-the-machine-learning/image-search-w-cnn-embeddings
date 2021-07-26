import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys

import albumentations as alb
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.append('../src/')
import aws_helper as awsh


# GLOBAL VARIABLES ================================================= #
data_split_to_use = 'train'
with open('../dataset/categories.json', 'r') as f:
    categories_of_interest = json.load(f)

# if 'api' is passed, then all images are retrieved from API
# otherwise 'local' means all images will be fetched from local machine
fetch_imgs_locally_or_api = 'local'
local_img_directory = '../data/raw/train/train2014/'

num_cpus = 4
batch_size = 250

s3_bucket = None
s3_key_prefix = 'coco_train_np_imgs'
local_np_dir = '../data/numpy_images/train/'

# TRANSFORM FUNCTIONS TO RESIZE & NORMALIZE ======================= #
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


# WRAPPER MAIN FUNCTION ========================================== #
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def main():
    # TODO: Write code to efficiently fetch from API
    if fetch_imgs_locally_or_api == 'local':
        print('Getting filepaths from local directory')
        filenames = os.listdir(local_img_directory)
        filepaths = [f"{local_img_directory}{fn}" for fn in tqdm(filenames)]

    i = 0
    while i < len(filenames):
        print(f"images: {i} to {i + batch_size}")
        print('Converting images to np arrays')
        cv2_imgs = [cv2.imread(fp) for fp in filepaths[i: i + batch_size]]
        np_images = [cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB) for cv2_img in tqdm(cv2_imgs)]

        pool = mp.Pool(processes=num_cpus)
        print('Resizing & padding images...')
        resized = [pool.apply(resize_and_pad_img, args=([np_img])) for np_img in tqdm(np_images)]
        print('Normalizing images...')
        normed = [pool.apply(normalize_img, args=([res])) for res in tqdm(resized)]
        pool.close()

        print('Saving numpy images locally into temp directory...')
        if not os.path.exists(local_np_dir):
            os.makedirs(local_np_dir)

        filename_slice = filenames[i: i + batch_size]
        np_filenames = [f"{fn.split('.')[0]}.np" for fn in filename_slice]
        process_paths = [f"{local_np_dir}{npfn}" for npfn in tqdm(np_filenames)]

        for np_img, np_filepath in zip(normed, process_paths):
            with open(np_filepath, 'wb') as p:
                np.save(p, np_img, allow_pickle=True)

        if s3_bucket is not None:
            print('Writing files to s3 bucket...')
            for path in tqdm(process_paths):
                np_filename = path.split('/')[-1]
                np_img = np.load(path)
                s3_keypath = f"{s3_key_prefix}/{np_filename}"
                awsh.upload_np_to_s3(s3_bucket, s3_keypath, np_img)

            print('Resetting ./tmp/ directory')
            shutil.rmtree('./tmp/')
            
        i += batch_size


if __name__ == "__main__":
    main()
