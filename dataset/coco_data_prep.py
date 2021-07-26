import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys

import albumentations as alb
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append('../utils/')
import aws_helper as awsh


# GLOBAL VARIABLES ================================================= #
data_split_to_use = 'train'
with open('../dataset/categories.json', 'r') as f:
    categories_of_interest = json.load(f)

# if 'api' is passed, then all images are retrieved from API
# otherwise 'local' means all images will be fetched from local machine
fetch_imgs_locally_or_api = 'local'
local_img_directory = '../data/raw/train/train2014/'

num_cpus = 32
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


def jpg_image_to_np_array(jpg_filepath) -> np.ndarray:
    """
    :param jpg_filepath: filepath for a single jpg image
    :return: np.array representation of image
    """
    cv2_img = cv2.imread(jpg_filepath)
    np_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return np_img


class COCODataset(Dataset):

    def __init__(self, data_dir: str, sample_ratio: int = None, randomize: bool = True):
        """
        :data_dir: data directory where images are stored (should be training or validation directory)
        :sample_ratio: if training, a % of the total training/validation data you wish to try your model on
            default is None, which means use all of the training or validation images in the specified data_dir
        :randomize: default is True
            when a sample_ratio is not None, the parameter to determine whether or not to just take top K images
            as the sample, OR to randomly select a subset of images from the data_dir
        """
        np.random.seed(42)
        self.data_dir = data_dir.rstrip('/')

        rand_file = np.random.choice(os.listdir(self.data_dir))
        filetype = rand_file.split('/')[-1].split('.')[-1]

        if filetype == 'np':
            self.filetype = 'binary'
        elif filetype == 'jpg' or filetype == 'jpeg':
            self.filetype = 'raw'
        else:
            raise ValueError("data_dir does not hold acceptable image file type, must be either `.np`, `.jpg`, or `.jpeg`")

        filepaths = [f"{self.data_dir}/{fn}" for fn in os.listdir(self.data_dir)]

        if self.filetype == 'binary':
            if sample_ratio is not None:
                num_imgs = int(sample_ratio * len(os.listdir(self.data_dir)))

                if randomize:
                    img_filepaths = np.random.choice(filepaths, num_imgs)
                else:
                    img_filepaths = filepaths[:num_imgs]

                self.dataset = [np.load(fp) for fp in tqdm(img_filepaths)]
            else:
                self.dataset = [np.load(fp) for fp in filepaths]

        elif self.filetype == 'jpg' or self.filetype == 'jpeg':
            num_cpus = mp.cpu_count()
            pool = mp.Pool(processes=num_cpus)
            print('Converting jpg images to np images')
            np_imgs = [pool.apply(jpg_image_to_np_array, args=([jpg_file])) for jpg_file in tqdm(os.listdir(self.data_dir))]
            print('Resizing and padding np images')
            resized = [pool.apply(resize_and_pad_img, args=([np_img])) for np_img in tqdm(np_images)]
            print('Normalizing images')
            normed = [pool.apply(normalize_img, args=([res])) for res in tqdm(resized)]
            pool.close()

        else:
            raise ValueError('file type in data_dir must be np or jpg.')

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


# WRAPPER MAIN FUNCTION ========================================== #
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


# main() function is to convert ALL jpeg images to numpy arrays
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


# if __name__ == "__main__":
#     main()
