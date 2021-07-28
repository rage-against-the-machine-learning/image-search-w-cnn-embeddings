import json
import multiprocessing as mp
import os
import shutil
import sys

import albumentations as alb
import cv2
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config_dataset
sys.path.append('../')
from utils import aws_helper as awsh


# GLOBAL VARIABLES ================================================= #
data_split_to_use = 'train'

with open ('../dataset/categories.json', 'r') as j:
    desired_categories = json.load(j)
with open ('../dataset/imgs_by_supercategory.json', 'r') as f:
    imgid_by_supercat = json.load(f)
# flatten to a list of imgids
desired_img_ids = list(set([ii for img_id in list(imgid_by_supercat.values()) for ii in img_id]))

# if 'api' is passed, then all images are retrieved from API
# otherwise 'local' means all images will be fetched from local machine
fetch_imgs_locally_or_api = 'local'

train_jpg_data_dir = '../data/raw/train/train2014/'
train_np_data_dir = '../data/numpy_imgs/train_subset/'
train_annot_path = '../data/raw/train/annotations/instances_train2014.json'

num_cpus = 32
batch_size = 250

s3_bucket = None
s3_key_prefix = 'coco_train_np_imgs'
local_np_dir = '../data/numpy_images/train/'

with open ('../dataset/imgs_by_supercategory.json', 'r') as f:
    imgid_by_supercat = json.load(f)


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
    def __init__(self,
                 np_img_data_dir,
                 annot_filepath,
                 sample_ratio: float = None):

        self.np_img_data_dir = np_img_data_dir

        self.sample_ratio = sample_ratio
        self.coco = COCO(annot_filepath)

        # All possible image ids
        all_train_img_ids = list(self.coco.imgs.keys())
        # Filter down to the image ids applicable to our supercategories
        self.ids = [ii for ii in all_train_img_ids if ii in desired_img_ids]

        if self.sample_ratio is None:
            pass
        else:
            self.ids = list(np.random.choice(self.ids, int(self.sample_ratio * len(all_train_img_ids))))

    def __getitem__(self, index):
        coco = self.coco

        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs([img_id])[0]['file_name']
        np_path = path.split('.')[0] + '.np'
        img = np.load(os.path.join(self.np_img_data_dir, np_path))

        return img, target

    def __len__(self):
        return len(self.ids)


def load_data (coco_dataset:COCODataset, batch_size: int, dataloader_params: dict = config_dataset.dataloader_params):
    dataloader_params.update({'batch_size': batch_size})
    dataloader_obj = DataLoader(coco_dataset, **dataloader_params)
    return dataloader_obj


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
