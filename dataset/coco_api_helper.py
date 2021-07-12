import os

import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io


# Set the filepaths for your annotation files in config_dataset.py
import config_dataset as config


# ALL the annotation files hold the same supercategories
coco_objects = {
    'train': COCO(config.train_annotation_filepath),
    'valid': COCO(config.valid_annotation_filepath),
    'test': COCO(config.test_annotation_filepath)
}


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def get_img_w_these_objects(desired_categories: list, data_subset: str, local_v_url: str = 'local') -> dict:
    """
    Fetch images containing the categories specified, if desired categories > 1 then images will hold all those labels
    :param desired_categories: list of categories or labels in images
    :param data_subset: any of ['train', 'valid', 'test']
        train_coco if you want images from training data
        valid_coco if you want images from validation subset...
    :param local_v_url: where to retrieve mages from
        local means from local machine
        url means download from coco url (via API)
    :return: dict {img_id: img as np array}
    """
    coco_obj = coco_objects.get(data_subset)
    category_ids = coco_obj.getCatIds(catNms=desired_categories)
    image_ids = coco_obj.getImgIds(catIds=category_ids)
    print(f"{len(image_ids)} image satisfy the intersection of listed categories.")

    if local_v_url == 'local':
        filenames = [coco_obj.loadImgs(image_ids[i])[0]['file_name'] for i in range(len(image_ids))]
        img_locs = [find(fn, config.img_filedir.get(data_subset)) for fn in filenames]
    else:
        img_locs = [coco_obj.loadImgs(image_ids[i])[0]['coco_url'] for i in range(len(image_ids))]

    images_by_imgid = {}
    for img_id, loc in zip(image_ids, img_locs):
        images_by_imgid[img_id] = io.imread(loc)
    return images_by_imgid

