"""
Functions to calculate precision & recall 
Compatible with Nearest Neighbors algorithm
Reference: https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54
"""
from collections import Counter
from operator import itemgetter

import numpy as np
import torch


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
