import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.linear_model


def plot_alpha_vs_r2(alphas, train_R2s, train_r2s, val_R2s, val_r2s, title=None):
    '''
    Args
    - alphas: list of float
    - train_/val_R2s: list of R^2 values
    - train_/val_r2s: list of r^2 values
    - title: str
    '''
    fig = plt.subplots(1, 1, figsize=[10, 4], constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.semilogx(alphas, train_R2s, label='train R^2', marker='.', markersize=6)
    ax.semilogx(alphas, train_r2s, label='train r^2', marker='.', markersize=6)
    ax.semilogx(alphas, val_R2s, label='val R^2', marker='.', markersize=6)
    ax.semilogx(alphas, val_r2s, label='val r^2', marker='.', markersize=6)
    ax.set_xlabel('alpha')
    ax.set_ylabel('score')
    min_r2 = np.nanmin([train_R2s, train_r2s, val_R2s, val_r2s])
    max_r2 = np.nanmax([train_R2s, train_r2s, val_R2s, val_r2s])
    ax.set_ylim(bottom=max(-1, min_r2) - 0.05, top=max_r2 + 0.05)
    if title is not None:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()
    plt.show()


def train_linear_model(train_X, train_y, val_X, val_y,
                       linear_model=sklearn.linear_model.Ridge,
                       plot_alphas=False, optimize='r2'):
    '''
    Args
    - train_X: np.array, shape [train_num_examples, num_features]
    - train_y: np.array, shape [train_num_examples]
    - val_X: np.array, shape [val_num_examples, num_features]
    - val_y: np.array, shape [val_num_examples]
    - linear_model: sklearn.linear_model
    - plot_alphas: bool, whether to plot alphas
    - optimize: str, one of ['r2', 'R2']

    Returns
    - best_model: sklearn.linear_model, learned model
    - best_train_preds: np.array, shape [train_num_examples], output of best_model on train_X
    - best_val_preds: np.array, shape [val_num_examples], output of best_model on val_X
    '''
    assert optimize in ['r2', 'R2']

    alphas = 2**np.arange(-5, 40, 0.5)
    train_R2s = np.zeros_like(alphas)
    train_r2s = np.zeros_like(alphas)
    val_R2s = np.zeros_like(alphas)
    val_r2s = np.zeros_like(alphas)
    best_model = None
    best_train_preds = None
    best_val_preds = None

    for i, alpha in enumerate(alphas):
        model = linear_model(alpha=alpha)
        model.fit(X=train_X, y=train_y)

        train_preds = model.predict(train_X)
        if plot_alphas:
            train_R2s[i] = sklearn.metrics.r2_score(y_true=train_y, y_pred=train_preds)
            train_r2s[i] = scipy.stats.pearsonr(train_y, train_preds)[0] ** 2

        val_preds = model.predict(val_X)
        val_R2 = sklearn.metrics.r2_score(y_true=val_y, y_pred=val_preds)
        val_r2 = scipy.stats.pearsonr(val_y, val_preds)[0] ** 2

        if (best_model is None) \
        or (optimize == 'r2' and val_r2 > np.max(val_r2s)) \
        or (optimize == 'R2' and val_R2 > np.max(val_R2s)):
            best_model = model
            best_val_preds = val_preds
            best_train_preds = train_preds

        val_R2s[i] = val_R2
        val_r2s[i] = val_r2

    if plot_alphas:
        best_index = np.argmax(val_r2s)
        print('best alpha: {:e}'.format(alphas[best_index]))
        plot_alpha_vs_r2(alphas, train_R2s, train_r2s, val_R2s, val_r2s)

    return best_model, best_train_preds, best_val_preds


def train_linear_logo(features, labels, group_labels, cv_groups, test_groups,
                      linear_model=sklearn.linear_model.Ridge,
                      plot=True, group_names=None, return_weights=False):
    '''Leave-one-group-out cross-validated training of a linear model.

    Args
    - features: np.array, shape [N, D]
    - labels: np.array, shape [N]
    - group_labels: np.array, shape [N], dtype np.int32
    - cv_groups: list of int, labels of groups to use for LOGO-CV
    - test_groups: list of int, labels of groups to test on
    - linear_model: sklearn.linear_model
    - plot: bool, whether to plot MSE as a function of alpha
    - group_names: list of str, names of the groups, only used when plotting
    - return_weights: bool, whether to return the final trained model weights

    Returns
    - test_preds: np.array, predictions on indices from test_groups
    - coefs: np.array, shape [D] (only returned if return_weights=True)
    - intercept: float (only returned if return_weights=True)
    '''
    cv_indices = np.isin(group_labels, cv_groups).nonzero()[0]
    test_indices = np.isin(group_labels, test_groups).nonzero()[0]

    X = features[cv_indices]
    y = labels[cv_indices]
    groups = group_labels[cv_indices]

    alphas = 2**np.arange(-5, 35, 3.0)
    preds = np.zeros([len(alphas), len(cv_indices)], dtype=np.float64)
    group_mses = np.zeros([len(alphas), len(cv_groups)], dtype=np.float64)
    leftout_group_labels = np.zeros(len(cv_groups), dtype=np.int32)
    logo = sklearn.model_selection.LeaveOneGroupOut()

    for i, alpha in enumerate(alphas):
        # set random_state for deterministic data shuffling
        model = linear_model(alpha=alpha, random_state=123)

        for g, (train_indices, val_indices) in enumerate(logo.split(X, y, groups)):
            train_X, val_X = X[train_indices], X[val_indices]
            train_y, val_y = y[train_indices], y[val_indices]
            model.fit(X=train_X, y=train_y)
            val_preds = model.predict(val_X)
            preds[i, val_indices] = val_preds
            group_mses[i, g] = np.mean((val_preds - val_y) ** 2)
            leftout_group_labels[g] = groups[val_indices[0]]

    mses = np.mean((preds - y) ** 2, axis=1)  # shape [num_alphas]

    if plot:
        h = max(3, len(group_names) * 0.2)
        fig, ax = plt.subplots(1, 1, figsize=[h*2, h], constrained_layout=True)
        for g in range(len(cv_groups)):
            group_name = group_names[leftout_group_labels[g]]
            ax.scatter(x=alphas, y=group_mses[:, g], label=group_name,
                       c=[cm.tab20.colors[g % 20]])
        ax.plot(alphas, mses, 'g-', label='Overall val mse')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Left-out Group')
        ax.set_xlabel('alpha')
        ax.set_ylabel('mse')
        ax.set_xscale('log')
        ax.grid(True)
        plt.show()

    best_alpha = alphas[np.argmin(mses)]
    best_model = linear_model(alpha=best_alpha)
    best_model.fit(X=X, y=y)
    test_preds = best_model.predict(features[test_indices])

    best_val_mse = np.min(mses)
    test_mse = np.mean((test_preds - labels[test_indices]) ** 2)
    print(f'best val mse: {best_val_mse:.3f}, best alpha: {best_alpha}, test mse: {test_mse:.3f}')

    if not return_weights:
        return test_preds
    else:
        coefs = best_model.coef_
        intercept = best_model.intercept_
        return test_preds, coefs, intercept
