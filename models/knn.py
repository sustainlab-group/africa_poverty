from utils.analysis import evaluate

import os
import time

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.spatial
import sklearn
from sklearn.neighbors import KNeighborsRegressor


def train_knn_logo(features, labels, group_labels, cv_groups, test_groups,
                   weights=None, plot=True, group_names=None, distance_metric='manhattan'):
    '''Leave-one-group-out cross-validated training of a KNN model.

    Args
    - features: np.array, shape [N, D]
        each feature dim should be normalized to 0 mean, unit variance
    - labels: np.array, shape [N]
    - group_labels: np.array, shape [N], type np.int32
    - cv_groups: list of int, labels of groups to use for LOGO-CV
    - test_groups: list of int, labels of groups to test on
    - weights: np.array, shape [N]
    - plot: bool, whether to plot MSE as a function of k
    - group_names: list of str, names of the groups, only used when plotting
    - distance_metric: str, see sklearn.neighbors.DistanceMetric

    Returns
    - test_preds: np.array, predictions on indices from test_groups
    '''
    cv_indices = np.isin(group_labels, cv_groups).nonzero()[0]
    test_indices = np.isin(group_labels, test_groups).nonzero()[0]

    X = features[cv_indices]
    y = labels[cv_indices]
    groups = group_labels[cv_indices]
    w = None if weights is None else weights[cv_indices]

    ks = 2 ** np.arange(0, 11)  # 1 to 1024
    preds = np.ones([len(ks), len(cv_indices)], dtype=np.float64) * np.nan
    group_mses = np.ones([len(ks), len(cv_groups)], dtype=np.float64) * np.nan
    leftout_group_labels = np.zeros(len(cv_groups), dtype=np.int32)
    logo = sklearn.model_selection.LeaveOneGroupOut()

    for i, k in enumerate(ks):
        model = KNeighborsRegressor(k, metric=distance_metric)

        for g, (train_indices, val_indices) in enumerate(logo.split(X, groups=groups)):
            if len(train_indices) < k:
                break
            train_X, val_X = X[train_indices], X[val_indices]
            train_y, val_y = y[train_indices], y[val_indices]
            val_w = None if w is None else w[val_indices]

            # assign each unique input training value the same training label
            if len(train_X.shape) == 1:  # scalars
                u = np.unique(train_X)
                new_train_y = np.zeros_like(train_y)
                for value in u:
                    mask = (train_X == value)
                    new_train_y[mask] = np.mean(train_y[mask])
                train_y = new_train_y

            model.fit(X=train_X, y=train_y)
            val_preds = model.predict(val_X)
            preds[i, val_indices] = val_preds
            group_mses[i, g] = np.average((val_preds - val_y) ** 2, weights=val_w)
            leftout_group_labels[g] = groups[val_indices[0]]

    mses = np.average((preds - y) ** 2, axis=1, weights=w)  # shape [K]

    if plot:
        h = max(3, len(group_names) * 0.2)
        fig, ax = plt.subplots(1, 1, figsize=(h*2, h), constrained_layout=True)
        for g, group_label in enumerate(leftout_group_labels):
            group_name = group_names[group_label]
            ax.scatter(x=ks, y=group_mses[:, g], label=group_name,
                       c=[cm.tab20.colors[g % 20]])
        ax.plot(ks, mses, 'g-', label='Overall val mse')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Left-out Group')
        ax.set(xlabel='k', ylabel='mse')
        ax.set_xscale('log')
        ax.grid(True)
        plt.show()

    best_k = ks[np.argmin(mses)]
    best_model = KNeighborsRegressor(best_k)
    best_model.fit(X=X, y=y)
    test_X, test_y = features[test_indices], labels[test_indices]
    test_preds = best_model.predict(test_X)

    best_val_mse = np.min(mses)
    test_w = None if weights is None else weights[test_indices]
    test_mse = np.average((test_preds - test_y) ** 2, weights=test_w)
    print(f'best val mse: {best_val_mse:.3f}, best k: {best_k}, test mse: {test_mse:.3f}')
    return test_preds


def knn_cv(features, labels, group_labels, group_names, savedir=None, weights=None,
           do_plot=False, subset_indices=None, subset_name=None, save_dict=None,
           distance_metric='manhattan'):
    '''
    For every fold F (the test fold):
      1. uses leave-one-fold-out CV on all other folds
         to tune KNN k parameter
      2. using best k, trains KNN model on all folds except F
      3. runs trained ridge model on F

    Saves predictions for each fold on test.
        savedir/test_preds_{subset_name}.npz if subset_name is given
        savedir/test_preds.npz otherwise

    Args
    - features: either a dict or np.array
        - if dict: group_name => np.array, shape [N, D]
            features to train on for a given test group
        - otherwise, just a single np.array, shape [N, D]
    - labels: np.array, shape [N]
    - group_labels: np.array, shape [N], type int
    - group_names: list of str, a group_label of X corresponds to group_names[X]
    - savedir: str, path to directory to save predictions
    - weights: np.array, shape [N], optional
    - do_plot: bool, whether to plot alpha vs. mse curve for 1st fold
    - subset_indices: np.array, indices of examples to include for both
        training and testing
    - subset_name: str, name of the subset
    - save_dict: dict, str => np.array, saved with test preds npz file
    - distance_metric: str, see sklearn.neighbors.DistanceMetric

    Returns
    - test_preds: np.array, shape [N]
    '''
    N = len(labels)
    if isinstance(features, np.ndarray):
        features = {f: features for f in group_names}
    for f in group_names:
        assert len(features[f]) == N

    if save_dict is None:
        save_dict = {}
    else:
        save_dict = dict(save_dict)  # make a copy

    if subset_indices is None:
        assert subset_name is None
        filename = 'test_preds.npz'
    else:
        assert subset_name is not None
        features = {f: feats[subset_indices] for f, feats in features.items()}
        labels = labels[subset_indices]
        group_labels = group_labels[subset_indices]

        filename = f'test_preds_{subset_name}.npz'
        for key in save_dict:
            save_dict[key] = save_dict[key][subset_indices]

    if savedir is not None:
        npz_path = os.path.join(savedir, filename)
        assert not os.path.exists(npz_path)

    test_preds = np.zeros_like(labels, dtype=np.float32)
    for i, f in enumerate(group_names):
        print('Group:', f)
        test_mask = (group_labels == i)
        if np.sum(test_mask) == 0:
            print(f'no examples corresponding to group {f} were found')
            continue
        test_preds[test_mask] = train_knn_logo(
            features=features[f],
            labels=labels,
            group_labels=group_labels,
            cv_groups=[x for x in range(len(group_names)) if x != i],
            test_groups=[i],
            weights=weights,
            plot=do_plot,
            group_names=group_names,
            distance_metric=distance_metric)

        # only plot the curve for the first group
        do_plot = False

    # save preds on the test set
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)

        # build up save_dict
        if 'labels' in save_dict:
            assert np.array_equal(labels, save_dict['labels'])
        save_dict['labels'] = labels
        if weights is not None:
            save_dict['weights'] = weights
        save_dict['test_preds'] = test_preds

        print('saving test preds to:', npz_path)
        np.savez_compressed(npz_path, **save_dict)

    return test_preds


def train_knn_logo_opt(dists, features, labels, group_labels, cv_groups, test_groups,
                       weights=None, plot=True, group_names=None):
    '''Leave-one-group-out cross-validated training of a KNN model.

    Similar to train_knn_logo(), but uses a pre-computed distance matrix.

    Args
    - dists: np.array, shape [N, N], precomputed distance matrix
    - features: np.array, shape [N, D]
    - labels: np.array, shape [N]
    - group_labels: np.array, shape [N], type np.int32
    - cv_groups: list of int, labels of groups to use for LOGO-CV
    - test_groups: list of int, labels of groups to test on
    - weights: np.array, shape [N]
    - plot: bool, whether to plot MSE as a function of k
    - group_names: list of str, names of the groups, only used when plotting

    Returns
    - test_preds: np.array, predictions on indices from test_groups
    '''
    cv_indices = np.isin(group_labels, cv_groups).nonzero()[0]
    test_indices = np.isin(group_labels, test_groups).nonzero()[0]

    dists_cv = dists[np.ix_(cv_indices, cv_indices)]
    X = features[cv_indices]
    y = labels[cv_indices]
    groups = group_labels[cv_indices]
    w = None if weights is None else weights[cv_indices]

    ks = 2 ** np.arange(0, 11)  # 1 to 1024
    preds = np.ones([len(ks), len(cv_indices)], dtype=np.float64) * np.nan
    group_mses = np.ones([len(ks), len(cv_groups)], dtype=np.float64) * np.nan
    leftout_group_labels = np.zeros(len(cv_groups), dtype=np.int32)
    logo = sklearn.model_selection.LeaveOneGroupOut()

    for g, (train_indices, val_indices) in enumerate(logo.split(X, groups=groups)):
        leftout_group_labels[g] = groups[val_indices[0]]

        train_X, train_y = X[train_indices], y[train_indices]
        val_y = y[val_indices]
        val_w = None if w is None else w[val_indices]

        # assign each unique input training value the same training label
        if len(train_X.shape) == 1:  # scalars
            u = np.unique(train_X)
            new_train_y = np.zeros_like(train_y)
            for value in u:
                mask = (train_X == value)
                new_train_y[mask] = np.mean(train_y[mask])
            train_y = new_train_y

        nearest_indices = np.argsort(dists_cv[np.ix_(val_indices, train_indices)], axis=1)

        for i, k in enumerate(ks):
            if len(train_indices) < k:
                break

            val_preds = np.mean(train_y[nearest_indices[:, :k]], axis=1)
            preds[i, val_indices] = val_preds
            group_mses[i, g] = np.average((val_preds - val_y) ** 2, weights=val_w)

    mses = np.average((preds - y) ** 2, axis=1, weights=w)  # shape [K]

    if plot:
        h = max(3, len(group_names) * 0.2)
        fig, ax = plt.subplots(1, 1, figsize=(h*2, h), constrained_layout=True)
        for g, group_label in enumerate(leftout_group_labels):
            group_name = group_names[group_label]
            ax.scatter(x=ks, y=group_mses[:, g], label=group_name,
                       c=[cm.tab20.colors[g % 20]])
        ax.plot(ks, mses, 'g-', label='Overall val mse')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Left-out Group')
        ax.set(xlabel='k', ylabel='mse')
        ax.set_xscale('log')
        ax.grid(True)
        plt.show()

    best_k = ks[np.argmin(mses)]

    # assign each unique input training value the same training label
    if len(X.shape) == 1:  # scalars
        u = np.unique(X)
        new_y = np.zeros_like(y)
        for value in u:
            mask = (X == value)
            new_y[mask] = np.mean(y[mask])
        y = new_y

    nearest_indices = np.argsort(dists[np.ix_(test_indices, cv_indices)], axis=1)
    test_preds = np.mean(y[nearest_indices[:, :best_k]], axis=1)

    best_val_mse = np.min(mses)
    test_y = labels[test_indices]
    test_w = None if weights is None else weights[test_indices]
    test_mse = np.average((test_preds - test_y) ** 2, weights=test_w)
    print(f'best val mse: {best_val_mse:.3f}, best k: {best_k}, test mse: {test_mse:.3f}')
    return test_preds


def knn_cv_opt(features, labels, group_labels, group_names, savedir=None, weights=None,
               do_plot=False, subset_indices=None, subset_name=None, save_dict=None,
               distance_metric='cityblock'):
    '''Similar to knn_cv(), but pre-computes a distance matrix to use for all folds.

    For every fold F (the test fold):
      1. uses leave-one-fold-out CV on all other folds
         to tune KNN k parameter
      2. using best k, trains KNN model on all folds except F
      3. runs trained ridge model on F

    Saves predictions for each fold on test.
        savedir/test_preds_{subset_name}.npz if subset_name is given
        savedir/test_preds.npz otherwise

    Args
    - features: np.array, shape [N, D]
    - labels: np.array, shape [N]
    - group_labels: np.array, shape [N], type int
    - group_names: list of str, a group_label of X corresponds to group_names[X]
    - savedir: str, path to directory to save predictions
    - weights: np.array, shape [N], optional
    - do_plot: bool, whether to plot alpha vs. mse curve for 1st fold
    - subset_indices: np.array, indices of examples to include for both
        training and testing
    - subset_name: str, name of the subset
    - save_dict: dict, str => np.array, saved with test preds npz file
    - distance_metric: str, see documentation for scipy.spatial.distance.pdist

    Returns
    - test_preds: np.array, shape [N]
    '''
    N = len(labels)
    assert len(features) == N
    assert len(group_labels) == N

    if save_dict is None:
        save_dict = {}
    else:
        save_dict = dict(save_dict)  # make a copy

    if subset_indices is None:
        assert subset_name is None
        filename = 'test_preds.npz'
    else:
        assert subset_name is not None
        features = features[subset_indices]
        labels = labels[subset_indices]
        group_labels = group_labels[subset_indices]

        filename = f'test_preds_{subset_name}.npz'
        for key in save_dict:
            save_dict[key] = save_dict[key][subset_indices]

    if savedir is not None:
        npz_path = os.path.join(savedir, filename)
        assert not os.path.exists(npz_path)

    print('Pre-computing distance matrix...', end='')
    start = time.time()
    dists = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(features, metric=distance_metric)
    )
    elapsed = time.time() - start
    print(f' took {elapsed:.2f} seconds.')

    test_preds = np.zeros_like(labels, dtype=np.float32)
    for i, f in enumerate(group_names):
        print('Group:', f)
        test_mask = (group_labels == i)
        if np.sum(test_mask) == 0:
            print(f'no examples corresponding to group {f} were found')
            continue
        test_preds[test_mask] = train_knn_logo_opt(
            dists=dists,
            features=features,
            labels=labels,
            group_labels=group_labels,
            cv_groups=[x for x in range(len(group_names)) if x != i],
            test_groups=[i],
            weights=weights,
            plot=do_plot,
            group_names=group_names)

        # only plot the curve for the first group
        do_plot = False

    evaluate(labels=labels, preds=test_preds, weights=weights, do_print=True, title='Pooled test preds')

    # save preds on the test set
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)

        # build up save_dict
        if 'labels' in save_dict:
            assert np.array_equal(labels, save_dict['labels'])
        save_dict['labels'] = labels
        if weights is not None:
            save_dict['weights'] = weights
        save_dict['test_preds'] = test_preds

        print('saving test preds to:', npz_path)
        np.savez_compressed(npz_path, **save_dict)

    return test_preds
