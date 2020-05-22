from collections import defaultdict
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import sklearn.cluster

sys.path.append('../')
from utils.geo_plot import plot_locs


def create_folds(locs, min_dist, dist_metric, fold_names, verbose=True,
                 plot_largest_clusters=0):
    '''Partitions locs into folds.

    Args
    - locs: np.array, shape [N, 2]
    - min_dist: float, minimum distance between folds
    - dist_metric: str, a valid distance metric accepted by sklearn.cluster.dbscan
    - fold_names: list of str, names of folds
    - verbose: bool
    - plot_largest_clusters: int, number of largest clusters to plot

    Returns
    - locs_to_indices: dict, maps (lat, lon) tuple to index in locs np.array
    - folds: dict, fold name => np.array of indices of locs belonging to that fold
    '''
    # there are duplicate locs => we want to cluster based on unique locs
    locs_to_indices = defaultdict(list)
    for i, loc in enumerate(locs):
        locs_to_indices[tuple(loc)].append(i)

    unique_locs = np.unique(locs, axis=0)  # get unique rows

    # any point within `min_dist` of another point belongs to the same cluster
    # - cluster_labels assigns a cluster index (0-indexed) to each loc
    # - a cluster label of -1 means that the point is an outlier
    _, cluster_labels = sklearn.cluster.dbscan(
        X=unique_locs, eps=min_dist, min_samples=2, metric=dist_metric)

    if verbose:
        _, unique_counts = np.unique(cluster_labels, return_counts=True)

        print('num clusters:', np.max(cluster_labels) + 1)  # clusters are 0-indexed
        print('max cluster size:', np.max(unique_counts[1:]))  # exclude outliers
        print('num outliers:', np.sum(cluster_labels == -1))

        fig, ax = plt.subplots(1, 1, figsize=[4, 2.5], constrained_layout=True)
        ax.hist(unique_counts[1:], bins=50)  # exclude outliers
        ax.set(xlabel='cluster size', ylabel='count')
        ax.set_yscale('log')
        ax.set_title('histogram of cluster sizes (excluding outliers)')
        ax.grid(True)
        plt.show()

    # mapping: cluster number => list of indices of points in that cluster
    # - if cluster label is -1 (outlier), then treat that unique loc as its own cluster
    neg_counter = -1
    clusters_dict = defaultdict(list)
    for loc, c in zip(unique_locs, cluster_labels):
        indices = locs_to_indices[tuple(loc)]
        if c < 0:
            c = neg_counter
            neg_counter -= 1
        clusters_dict[c].extend(indices)

    # sort clusters by descending cluster size
    sorted_clusters = sorted(clusters_dict.keys(), key=lambda c: -len(clusters_dict[c]))

    # plot the largest clusters
    for i in range(plot_largest_clusters):
        c = sorted_clusters[i]
        indices = clusters_dict[c]
        title = 'cluster {c}: {n} points'.format(c=c, n=len(indices))
        plot_locs(locs[indices], figsize=(4, 4), title=title)

    # greedily assign clusters to folds
    folds = {f: [] for f in fold_names}
    for c in sorted_clusters:
        # assign points in cluster c to smallest fold
        f = min(folds, key=lambda f: len(folds[f]))
        folds[f].extend(clusters_dict[c])

    for f in folds:
        folds[f] = np.sort(folds[f])

    return locs_to_indices, folds


def verify_folds(folds, locs, min_dist, dist_metric, max_index=None):
    '''Verifies that folds do not overlap.

    Args
    - folds: dict, fold name => np.array of indices of locs belonging to that fold
    - locs: np.array, shape [N, 2], each row is [lat, lon]
    - min_dist: float, minimum distance between folds
    - dist_metric: str, a valid distance metric accepted by sklearn.cluster.dbscan
    - max_index: int, all indices in range(max_index) should be included
    '''
    for fold, idxs in folds.items():
        assert np.all(np.diff(idxs) >= 0)  # check that indices are sorted

    # check that all indices are included
    if max_index is not None:
        assert np.array_equal(
            np.sort(np.concatenate(list(folds.values()))),
            np.arange(max_index))

    # check to ensure no overlap
    for a, b in itertools.combinations(folds.keys(), r=2):
        a_idxs = folds[a]
        b_idxs = folds[b]
        dists = scipy.spatial.distance.cdist(locs[a_idxs], locs[b_idxs], metric=dist_metric)
        assert np.min(dists) > min_dist
        print(a, b, np.min(dists))
