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
    
    cat_precision_at_k = cat_matches / at_k
    supercat_precision_at_k = supercat_matches / at_k
        
    return cat_precision_at_k, supercat_precision_at_k


def annoy_recall (annoy_obj, query_idx: int, idx2coco_map: dict,
                     n_neighbors: int = 10, at_k: int = 4):
    """
    :annoy_obj: trained annoy object fit w/ embeddings
    :query_idx: single embedding index
    :idx2coco_map: hashmap with keys as embedding index, values as the coco id
    :n_neighbors: number of images to retrieve
    :at_k: depth at which to calculate recall
    """
    
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
    
    cat_matches_at_k = np.sum([1 if l == ground_truth_label else 0 for l in cat_labels][1: at_k + 1])
    supercat_matches_at_k = np.sum([1 if l == ground_truth_supercat else 0 for l in supercat_labels][1: at_k + 1])
 
    cat_matches = np.sum([1 if l == ground_truth_label else 0 for l in cat_labels][1:])
    supercat_matches = np.sum([1 if l == ground_truth_supercat else 0 for l in supercat_labels][1:])
    
    # turnery operator to ensure no division by 0 takes place
    cat_recall_at_k = cat_matches / cat_matches if cat_matches != 0 else 0
    supercat_recall_at_k = supercat_matches / supercat_matches if supercat_matches != 0 else 0
        
    return cat_recall_at_k, supercat_recall_at_k


# NEAREST NEIGHBOR METRICS ========================================== #

def nearest_neighbors_precision(plotter, idx, nbrs, k=4, category=True, reduced_space=None):
    """
    :plotter: plotter object from plot_utils.py
    :idx: embedding index
    :nbrs: for nearest neighbors object
    :k: number of similar images to retrieve
    :category: if True then calculate on categories, otherwise calcuate at supercategory
    :reduced_space: embeddings file if embeddings were subsequently passed through dim reduction algo
    """
    if reduced_space is None:
        embeddings = plotter.embeddings
        m, n = embeddings.size()
    else:
        embeddings = reduced_space
        m, n = np.shape(embeddings)
        
    distances, indices = nbrs.kneighbors(embeddings[idx, :].reshape(1, n), nbrs.n_neighbors)
    retrieved_distances = distances[0][1:k+1]
    retrieved_indices = indices[0][1:k+1]
    
    if category == True:
        ground_truth = plotter.category_labels[idx]
        retrieved_labels = itemgetter(*retrieved_indices)(plotter.category_labels)
    else:
        ground_truth = plotter.supercategory_labels[idx]
        retrieved_labels = itemgetter(*retrieved_indices)(plotter.supercategory_labels)
        
    precision = np.array([1 if i == ground_truth else 0 for i in retrieved_labels]).sum() / (nbrs.n_neighbors-1)
    return precision

def nearest_neighbors_avg_precision(plotter, nbrs, query_idxs: list, k=4, category=True, reduced_space=None):
    """
    :plotter: plotter object from plot_utils.py
    :nbrs: for nearest neighbors object
    :query_idxs: embedding indexes corresponding to the retrieved images
    :category: if True then calculate on categories, otherwise calcuate at supercategory
    :reduced_space: embeddings IF they were subsequently passed through a dim reduction
    """
    avg_precision = 0
    for idx in query_idxs:
        avg_precision += nearest_neighbors_precision(plotter, idx, nbrs, k, category, reduced_space)
    avg_precision = avg_precision / len(query_idxs)
    return avg_precision


def nearest_neighbors_recall(plotter, idx, nbrs, k=4, category=True, reduced_space=None):
    """
    :plotter: plotter object from plot_utils.py
    :idx: a single embedding index,
    :nbrs: nearest neighbors insantiated & trained object
    :k: number of imagres to retrieve (Incl the query image) 
    :category: True if calculating at subcategory level, FALSE for super category
    :reduced_space: embeddings array 
    """
    if reduced_space is None:
        embeddings = plotter.embeddings
        m, n = embeddings.size()
    else:
        embeddings = reduced_space
        m, n = np.shape(embeddings)
    distances, indices = nbrs.kneighbors(embeddings[idx, :].reshape(1, n), nbrs.n_neighbors)
    retrieved_distances = distances[0][1:k+1]
    retrieved_indices = indices[0][1:k+1]
    
    if category == True:
        ground_truth = plotter.category_labels[idx]
        total_retrieved_labels = itemgetter(*indices[0][1:])(plotter.category_labels)
        retrieved_labels = itemgetter(*retrieved_indices)(plotter.category_labels)
        
    else:
        ground_truth = plotter.supercategory_labels[idx]
        total_retrieved_labels = itemgetter(*indices[0][1:])(plotter.supercategory_labels)
        retrieved_labels = itemgetter(*retrieved_indices)(plotter.supercategory_labels)

    relevant_labels = [i for i in total_retrieved_labels if i == ground_truth]
    recall = (np.array([1 if i == ground_truth else 0 for i in retrieved_labels]).sum() / len(relevant_labels)) if len(relevant_labels) != 0 else 0
    return recall

    
def avg_recall(plotter, nbrs, query_idxs, k=4, category=True, reduced_space=None):
    """
    :plotter: plotter object from plot_utils.py
    :query_idx: embedding indexes corresponding to the retrieved images
    :k: the # of images to retrieve (incl the query image)
    :category: if True then calculate on categories, otherwise calculate at supercategory
    :reduced_sapce: embeddings IF they were subsequently passed through dim reduction like PCA or TSNE
    """
    avg_recall = 0
    for idx in query_idxs:
        avg_recall += nearest_neighbors_recall(plotter, idx, nbrs, k, category, reduced_space)
    avg_recall = avg_recall / len(query_idxs)
    return avg_recall
    
