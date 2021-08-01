# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 07:23:51 2021

@author: micha
"""

# load and display instance annotations

import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import numpy as np
import itertools
from pycocotools.coco import COCO
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import pandas as pd

class EmbeddingPlotter:
    def __init__(self, annot_file: str, embeddings, idx2Img: dict):
        """
        :annot_file: annot file path 
        :embeddings: pytorch embedding files
        :idx2Img: the dictionary with key == embedding index, and value == coco image id
        """
        self.annot_file = annot_file
        self.coco = COCO(self.annot_file)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_name_dict = {cat['id']: cat['name'] for cat in cats}
        self.img_ids = list(idx2Img.values())
        self.annIds = self.coco.getAnnIds(self.img_ids)
        
        self.embeddings = embeddings
        self.idx2Img = idx2Img
        self.labels = []
        for k, v in self.idx2Img.items():
            ann_ids_to_append = self.coco.getAnnIds(self.idx2Img[k])
            anns_to_append = self.coco.loadAnns(ann_ids_to_append)
            self.labels.append(anns_to_append)
        
        self.category_labels= []
        self.supercategory_labels = []
        
        cats_dict = {cat['id']: cat for cat in cats}
        for label in self.labels:
            
            if len(label) > 0:
                categories = []
                supercategories = []
                
                for annotation in label:
                    categories.append(cats_dict[annotation['category_id']]['name'])
                    supercategories.append(cats_dict[annotation['category_id']]['supercategory'])
        
                # most common category among objects in the picture
                category, _ = Counter(categories).most_common(1)[0]
                supercategory, _ = Counter(supercategories).most_common(1)[0]
                
                self.category_labels.append(category)
                self.supercategory_labels.append(supercategory)
            else:
                self.category_labels.append(None)
                self.supercategory_labels.append(None)
        
    def plot_coco_images(self, 
                         img_ids,
                         imgs_per_row: int = 5,
                         show_labels=True,
                         show_annotations=False, 
                         title_suffix=None, 
                         figtitle=None,
                         ncols: int = None):
        """
        :img_ids: coco image ids 
        """
        coco = self.coco
        cat_id_to_name_dict = self.cat_id_to_name_dict
        
        ncols = imgs_per_row
        nrows = len(img_ids[0]) // imgs_per_row + 1
        
        if ncols is not None:
            ncols = ncols
        else:
            ncols = len(img_ids[0])
    
        fig = plt.figure(1, figsize=(int(nrows * imgs_per_row), ncols * imgs_per_row // nrows))
    
        for i, img_id in enumerate(itertools.chain(*img_ids)):
            ax = fig.add_subplot(nrows, ncols, i+1)
            img = coco.loadImgs(img_id)[0]['coco_url']
    
            ax.imshow(img, aspect='equal')
            ax.axis('off')
            
            if title_suffix is not None:
                title=f'{img_id}: {title_suffix[i // nrows, i % ncols]}'
            else:
                title = str(img_id)
            ax.set_title(title)
    
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)
            if show_annotations:
                coco.showAnns(anns)
    
            if show_labels:
                for ann in anns:
                    [x, y, _, _] = ann['bbox']
                    label = cat_id_to_name_dict[ann['category_id']]    
                    t = ax.annotate(label, (x,y), fontsize=12)
                    t.set_bbox(dict(facecolor='white', alpha=0.2, edgecolor='white'))
    
        plt.tight_layout()
        
        if figtitle:
            fig.suptitle(figtitle)
            plt.subplots_adjust(top=0.90)
        plt.show()
        
    def plot_neighbors(self, X_idx, nbrs, show_annotations=False):
        """
        :X_idx: the embedding 
        """
        idx2Img = self.idx2Img
        embeddings = self.embeddings
#         m, n = np.shape(embeddings)
        num_emb, emb_dim = np.shape(embeddings)
        img_id = idx2Img[X_idx]
        
        distances, indices = nbrs.kneighbors(embeddings[X_idx, :].reshape(1, emb_dim), nbrs.n_neighbors)
        # distances & indices shaped like this = [[elem1, elem2, ... elemN]]
        
        distances, indices = distances[0], indices[0]
        neighbors_ids = np.array(list(map(idx2Img.get, indices)))
        np.insert(neighbors_ids, 0, img_id)
        np.insert(distances, 0, 0)
        
        title_suffix = np.array([f'distance={distance:.2f}' for distance in distances])
        figtitle=f'Neighbors of {img_id}'
        
        
        self.plot_coco_images(neighbors_ids.reshape(len([img_id]), len(indices) // len([img_id])).tolist(), 
                         title_suffix=title_suffix.reshape(len([img_id]), len(indices)//len([img_id])),
                         show_annotations=show_annotations,
                         figtitle=figtitle,
                         ncols=nbrs.n_neighbors + 1)
        
    def plot_3d(self, X, title=None, columns=['x0', 'x1', 'x2'], **kwargs):
        labels = self.supercategory_labels
                
        df = pd.DataFrame(X, columns=columns)
        df['label'] = labels
        cmap = plt.get_cmap('gist_rainbow')
        unique_labels = list(set(labels))
        labels_dict = {label: i for i, label in enumerate(unique_labels)}
        num_colors = len(unique_labels)
        c = [cmap(1.*labels_dict[label]/num_colors) for label in labels]
    
        df['c'] = c
        df = df[~df['label'].isna()]
        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        
        for label, grp in df.groupby('label'):
            ax.scatter(
            xs=grp[columns[0]], 
            ys=grp[columns[1]], 
            zs=grp[columns[2]], 
            c=grp['c'], 
            label=label,
            **kwargs
        )
        
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.set_zlabel(columns[2])
        
        plt.legend(loc='best')
        if title:
            plt.title(title)
        plt.show()