"""
Functions to calculate precision & recall 
Compatible with Nearest Neighbors algorithm
Reference: https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54
"""
from collections import Counter
from operator import itemgetter
import sys

import numpy as np
import torch

sys.path.append('../dataset/')
import coco_api_helper


# GLOBAL VARIABLES ======================================= #

val_coco = coco_api_helper.coco_objects['valid']
category_id2name = {cat['id']: cat['name'] for cat in val_coco.loadCats(val_coco.getCatIds())}


# ANNOY METRICS ========================================== #

def annoy_precision (annoy_obj, query_idx: int, idx2coco_map: dict,
                     n_neighbors: int = 10, at_k: int = 4):
    
    closest = annoy_obj.get_nns_by_item(query_idx, n_neighbors)
    closest_at_k = closest[1: at_k + 1]
    
    cocoids = [idx2coco_map.get(idx) for idx in closest]
    annids = [val_coco.getAnnIds(cid) for cid in cocoids]
    
    ground_truth_annots = val_coco.loadAnns(annids[0])
    ground_truth_catids = [ann['category_id'] for ann in ground_truth_annots]  
    ground_truth_label = category_id2name.get(Counter(ground_truth_catids).most_common()[0][0])
    ground_truth_supercat = val_coco.cats.get(Counter(ground_truth_catids).most_common()[0][0])['supercategory']
        
    cat_labels = []
    supercat_labels = []
    
    for an in annids:
        cat_ids = []
        
        if len(an) > 0:
            loaded_annots = [val_coco.loadAnns(a) for a in an]
            cat_ids.append(Counter([l['category_id'] for loaded in loaded_annots for l in loaded]).most_common()[0][0])
            supercat_labels.append([val_coco.cats.get(Counter([l['category_id'] for loaded in loaded_annots for l in loaded]).most_common()[0][0])['supercategory']][0])
        else:
            return None 
        
        cat_labels.extend([category_id2name.get(cid) for cid in cat_ids])
    
    cat_matches = np.sum([1 if l == ground_truth_label else 0 for l in cat_labels][1: at_k + 1])
    supercat_matches = np.sum([1 if l == ground_truth_supercat else 0 for l in supercat_labels][1: at_k + 1])
    
    cat_precision_at_k = cat_matches / n_neighbors
    supercat_precision_at_k = supercat_matches / n_neighbors
        
    return cat_precision_at_k, supercat_precision_at_k



# NEAREST NEIGHBOR METRICS ========================================== #

def precision(plotter, idx, nbrs, category=True, reduced_space=None):
    """
    :plotter: plotter object from plot_utils.py
    :idx: embedding nidex
    :nbrs: for nearest neighbors object
    :category: if True then calculate on categories, otherwise calcuate at supercategory
    """
    if reduced_space is None:
        embeddings = plotter.embeddings
        m, n = embeddings.size()
    else:
        embeddings = reduced_space
        m, n = np.shape(embeddings)
        
    distances, indices = nbrs.kneighbors(embeddings[idx, :].reshape(1, n), 16)
    distances = distances[0][1:]
    indices = indices[0][1:]
    if category == True:
        ground_truth = plotter.category_labels[idx]
        retrieved_labels = itemgetter(*indices)(plotter.category_labels)
    else:
        ground_truth = plotter.supercategory_labels[idx]
        retrieved_labels = itemgetter(*indices)(plotter.supercategory_labels)
        
    precision = np.array([1 if i == ground_truth else 0 for i in retrieved_labels]).sum() / len(retrieved_labels)
    return precision


def avg_precision(plotter, nbrs, query_idxs, category=True, reduced_space=None):
    """
    :plotter: plotter object from plot_utils.py
    :nbrs: for nearest neighbors object
    :query_idxs: embedding indexes corresponding to the retrieved images
    :category: if True then calculate on categories, otherwise calcuate at supercategory
    :reduced_space: embeddings IF they were subsequently passed through a dim reduction
    """
    avg_precision = 0
    for idx in query_idxs:
        avg_precision += precision(plotter, idx, nbrs, category, reduced_space)
    avg_precision = avg_precision / len(query_idxs)
    return avg_precision


def recall(plotter, idx, nbrs, category=True, reduced_space=None):
    """
    :plotter: plotter object from plot_utils.py
    :idx: embedding nidex
    :nbrs: for nearest neighbors object
    :category: if True then calculate on categories, otherwise calcuate at supercategory
    """
    if reduced_space is None:
        embeddings = plotter.embeddings
        m, n = embeddings.size()
    else:
        embeddings = reduced_space
        m, n = np.shape(embeddings)
        
    distances, indices = nbrs.kneighbors(embeddings[idx, :].reshape(1, n), 16)
    distances = distances[0][1:]
    indices = indices[0][1:]

    if category == True:
        ground_truth = plotter.category_labels[idx]
        retrieved_labels_at_k = itemgetter(*indices)(plotter.category_labels)[:nbrs.n_neighbors]
        retrieved_labels_all = itemgetter(*indices)(plotter.category_labels)
        num_rel_items_at_k = Counter(retrieved_labels_at_k)[ground_truth]
        num_rel_items_all = Counter(retrieved_labels_all)[ground_truth]
        
    else:
        ground_truth = plotter.supercategory_labels[idx]
        retrieved_labels_at_k = itemgetter(*indices)(plotter.supercategory_labels)[:nbrs.n_neighbors]
        retrieved_labels_all = itemgetter(*indices)(plotter.supercategory_labels)
        num_rel_items_at_k = Counter(retrieved_labels_at_k)[ground_truth]
        num_rel_items_all = Counter(retrieved_labels_all)[ground_truth]

    recall = num_rel_items_at_k / num_rel_items_all if num_rel_items_all != 0 else 0
    return recall


def avg_recall(plotter, nbrs, query_idxs, category=True, reduced_space=None):
    """
    :plotter: plotter object from plot_utils.py
    :nbrs: for nearest neighbors object
    :query_idxs: embedding indexes corresponding to the retrieved images
    :category: if True then calculate on categories, otherwise calcuate at supercategory
    :reduced_space: embeddings IF they were subsequently passed through a dim reduction
    """
    average_recall = []
    
    for idx in query_idxs:
        average_recall.append(recall(plotter, idx, nbrs, category, reduced_space))
    avg_recall = np.mean(average_recall)
    
    return avg_recall
