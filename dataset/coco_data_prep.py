import json
import multiprocessing as mp
import os
import shutil
import sys

import albumentations as alb
import cv2
import numpy as np
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import coco_api_helper
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
train_img_ids = list(set([ii for img_id in list(imgid_by_supercat.values()) for ii in img_id]))

# Validation image ids 
category_ids = [cat['id'] for cat in desired_categories]
valid_annot = coco_api_helper.coco_objects['valid']
val_img_ids = [valid_annot.getImgIds(catIds=[id]) for id in category_ids]
val_img_ids = list(set([i for ii in val_img_ids for i in ii]))

img_id_by_split = dict(
    train=train_img_ids,
    val=val_img_ids
)

# if 'api' is passed, then all images are retrieved from API
# otherwise 'local' means all images will be fetched from local machine
fetch_imgs_locally_or_api = 'local'

train_jpg_data_dir = '../data/raw/train/train2014/'
train_np_data_dir = '../data/numpy_imgs/train_subset/'
train_annot_path = '../data/raw/train/annotations/instances_train2014.json'

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


def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


class COCOAnnotationTransform(object):
    # https://github.com/amdegroot/ssd.pytorch/blob/master/data/coco.py
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")
                

class COCODataset(Dataset):
    def __init__(self,
                 data_split: str,
                 np_img_data_dir,
                 annot_filepath,
                 sample_ratio: float = None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """
        :data_split: is one of 'train' or 'val' and the corresponding img ids will be fetched
        :np_img_data_dir: local directory where you have unzipped np files saved 
            np files are the trasnformed images from s3 that were normalized/ resized/ padded
        :annot_filepath: filepath of the original coco dataset corresponding to the datasplit of your choosing
        :sample_ratio: specified float between 0 and 1 for the % of images from the data split you want to use
        :device: cpu or gpu
        """
        if data_split not in ['train', 'val']:
            raise ValueError("data_split param must be one of 'train', or 'val'")
        else:
            self.data_split = data_split
            
        self.device = device
        self.np_img_data_dir = np_img_data_dir

        self.sample_ratio = sample_ratio
        self.coco = COCO(annot_filepath)

        # All possible image ids
        all_img_ids = list(self.coco.imgs.keys())
        # Filter down to the image ids applicable to our supercategories
        img_ids_to_get = img_id_by_split.get(self.data_split)
        self.ids = [ii for ii in tqdm(all_img_ids) if ii in img_ids_to_get]

        if self.sample_ratio is None:
            pass
        else:
            self.ids = list(np.random.choice(self.ids, int(self.sample_ratio * len(all_img_ids)), replace=False))

    def __getitem__(self, index):
        coco = self.coco

        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs([img_id])[0]['file_name']
        np_path = path.split('.')[0] + '.np'
        img = np.load(os.path.join(self.np_img_data_dir, np_path))

        # Convert the image to tensor so it's compatible w/ pytorch data loader
        img = torch.Tensor(img.transpose(2,0,1)).to(device=self.device).float()

        return img, target

    def __len__(self):
        return len(self.ids)
    
    
def get_dataloader(dataset_obj,
                   batch_size: int=100,
                   device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                   loader_params:dict = config_dataset.dataloader_params):
    """Returns data loader for custom dataset.
    Args:
        :dataset_obj: dataset object returned from COCODataset class
        :batch_size: specified batch size, if you have memory errors while running, make this smaller
        :device: default is CUDA if there is GPU otherwise CPU
        :loader_params: default is found in config_dataset.py's dataloader_params dict
            can specify your own data loader params, but collate_fn must be `lambda x: x`
            specify:
                shuffle: True / False
                num_workers: number of parallel processes to run
                collate_fn: lambda x: x
    Returns:
        data_loader: data loader for COCODataset.
    """
    # data loader for custom dataset
    # this will return (imgs, targets) for each iteration
    loader_params.update({'batch_size': batch_size})
    
    data_loader = DataLoader(
        dataset=dataset_obj, 
        **loader_params,
    )
    return data_loader


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
