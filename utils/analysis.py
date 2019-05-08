import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import sklearn.metrics


def calc_score(labels, preds, metric, weights=None):
    '''
    See https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    for the weighted correlation coefficient formula.

    Args
    - labels: np.array, shape [N]
    - preds: np.array, shape [N]
    - score: str, one of ['r2', 'R2', 'mse', 'rank']
        - 'r2': (weighted) squared Pearson correlation coefficient
        - 'R2': (weighted) coefficient of determination
        - 'mse': (weighted) mean squared-error
        - 'rank': (unweighted) Spearman rank correlation coefficient
    - weights: np.array, shape [N]

    Returns: float
    '''
    if metric == 'r2':
        if weights is None:
            return scipy.stats.pearsonr(labels, preds)[0] ** 2
        else:
            mx = np.average(preds, weights=weights)
            my = np.average(labels, weights=weights)
            cov_xy = np.average((preds - mx) * (labels - my), weights=weights)
            cov_xx = np.average((preds - mx) ** 2, weights=weights)
            cov_yy = np.average((labels - my) ** 2, weights=weights)
            return cov_xy ** 2 / (cov_xx * cov_yy)
    elif metric == 'R2':
        return sklearn.metrics.r2_score(y_true=labels, y_pred=preds,
                                        sample_weight=weights)
    elif metric == 'mse':
        return np.average((labels - preds) ** 2, weights=weights)
    elif metric == 'rank':
        return scipy.stats.spearmanr(labels, preds)[0]
    else:
        raise ValueError(f'Unknown metric: "{metric}"')


def calc_r2(x, y):
    return calc_score(labels=x, preds=y, metric='r2')


def evaluate(labels, preds, weights=None, do_print=False, title=None):
    '''
    Args
    - labels: list of labels, length N
    - preds: list of preds, length N
    - weights: list of weights, length N
    - do_print: bool
    - title: str

    Returns: r^2, R^2, mse, rank_corr
    '''
    r2 = calc_score(labels=labels, preds=preds, metric='r2', weights=weights)
    R2 = calc_score(labels=labels, preds=preds, metric='R2', weights=weights)
    mse = calc_score(labels=labels, preds=preds, metric='mse', weights=weights)
    rank = calc_score(labels=labels, preds=preds, metric='rank', weights=weights)
    if do_print:
        if title is not None:
            print(f'{title}\t- ', end='')
        print(f'r^2: {r2:0.3f}, R^2: {R2:0.3f}, mse: {mse:0.3f}, rank: {rank:0.3f}')
    return r2, R2, mse, rank


def evaluate_df(df):
    '''Runs `evaluate` on a pandas DataFrame.

    Example usage
    >>> preds_df.groupby('bands').apply(evaluate_df)

    Args
    - df: pd.DataFrame, columns include ['preds', 'labels']

    Returns: pd.Series, index = ['r2', 'R2', 'mse', 'rank']
    '''
    r2, R2, mse, rank = evaluate(df['preds'], df['labels'])
    return pd.Series({'r2': r2, 'R2': R2, 'mse': mse, 'rank': rank})


def plot_predictions(labels, preds, title=None):
    '''
    Args
    - labels: list of labels, length n
    - preds: list of preds, length n
    - title: str
    '''
    with sns.axes_style('whitegrid'):
        g = sns.jointplot(x=labels, y=preds, kind='reg',
                          joint_kws={'scatter_kws': {'s': 5}})
    g.set_axis_labels('True Label', 'Predicted Label')
    if title is not None:
        g.fig.suptitle(title)
    plt.grid(b=True)
    xy_line = np.array([-2, 3])
    plt.plot(xy_line, xy_line, color='black')
    plt.show()


def plot_residuals(labels, preds, title=None):
    '''Plots residuals = preds - labels.

    Args
    - labels: np.array, shape [N]
    - preds: np.array, shape [N]
    - title: str, ie. 'train' or 'val'
    '''
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    flat_line = np.array([-2, 3])
    ax.plot(flat_line, np.zeros_like(flat_line), color='black')

    residuals = preds - labels
    ax.scatter(x=labels, y=residuals, label='residuals')
    ax.legend()
    ax.grid()
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('labels')
    ax.set_ylabel('residuals')


def sorted_scores(labels, preds, metric, sort='increasing'):
    '''
    Sorts (pred, label) datapoints by label using the given sorting direction,
    then calculates the chosen score over for the first k datapoints,
    for k = 1 to N.

    Args
    - labels: np.array, shape [N]
    - preds: np.array, shape [N]
    - metric: one of ['r2', 'R2', 'mse', 'rank']
    - sort: str, one of ['increasing', 'decreasing', 'random']

    Returns:
    - scores: np.array, shape [N]
    - labels_sorted: np.array, shape [N]
    '''
    if sort == 'increasing':
        sorted_indices = np.argsort(labels)
    elif sort == 'decreasing':
        sorted_indices = np.argsort(labels)[::-1]
    elif sort == 'random':
        sorted_indices = np.random.permutation(len(labels))
    else:
        raise ValueError(f'Unknown value for sort: {sort}')
    labels_sorted = labels[sorted_indices]
    preds_sorted = preds[sorted_indices]
    scores = np.zeros(len(labels))
    for i in range(1, len(labels) + 1):
        scores[i - 1] = calc_score(labels=labels_sorted[0:i], preds=preds_sorted[0:i], metric=metric)
    return scores, labels_sorted


def plot_label_vs_score(scores_list, labels_list, legends, metric, sort):
    '''
    Args
    - scores_list: list of length num_models, each element is np.array of shape [num_examples]
    - labels_list: list of length num_models, each element is np.array of shape [num_examples]
    - legends: list of str, length num_models
    - metric: str, metric by which the scores were calculated
    - sort: str, one of ['increasing', 'decreasing', 'random']
    '''
    num_models = len(scores_list)
    assert len(labels_list) == num_models
    assert len(legends) == num_models

    f, ax = plt.subplots(1, 1, figsize=[8, 6], constrained_layout=True)
    for i in range(num_models):
        ax.scatter(x=labels_list[i], y=scores_list[i], s=2)

    ax.set_xlabel('wealthpooled')
    ax.set_ylabel(metric)
    ax.set_title(f'Model Performance vs. cumulative {sort} label')
    ax.set_ylim(-0.05, 1.05)
    ax.grid()
    lgd = ax.legend(legends)
    for handle in lgd.legendHandles:
        handle.set_sizes([30.0])
    plt.show()


def plot_percdata_vs_score(scores_list, legends, metric, sort):
    '''
    Args
    - scores_list: list of length num_models, each element is np.array of shape [num_examples]
    - legends: list of str, length num_models
    - metric: str, metric by which the scores were calculated
    - sort: str, one of ['increasing', 'decreasing', 'random']
    '''
    num_models = len(scores_list)
    assert len(legends) == num_models

    f, ax = plt.subplots(1, 1, figsize=[8, 6], constrained_layout=True)
    for i in range(num_models):
        num_examples = len(scores_list[i])
        percdata = np.arange(1, num_examples + 1, dtype=np.float32) / num_examples
        ax.scatter(x=percdata, y=scores_list[i], s=2)
    ax.set_xlabel('% of data')
    ax.set_ylabel(metric)
    ax.set_title(f'Model Performance vs. % of data by {sort} label')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid()
    lgd = ax.legend(legends)
    for handle in lgd.legendHandles:
        handle.set_sizes([30.0])
    plt.show()


def chunk_vs_score(labels, preds, nchunks, metric):
    '''
    Args
    - labels: np.array, shape [N]
    - preds: np.array, shape [N]
    - nchunks: int
    - metric: str, one of ['r2', 'R2', 'mse', 'rank']

    Returns
    - scores: np.array, shape [nchunks]
    '''
    sorted_indices = np.argsort(labels)
    chunk_indices = np.array_split(sorted_indices, nchunks)  # list of np.array
    scores = np.zeros(nchunks)
    for i in range(nchunks):
        chunk_labels = labels[chunk_indices[i]]
        chunk_preds = preds[chunk_indices[i]]
        scores[i] = calc_score(labels=chunk_labels, preds=chunk_preds, metric=metric)
    return scores


def plot_chunk_vs_score(scores, legends, metric):
    '''
    Args
    - scores: np.array, shape [num_models, nchunks]
    - legends: list of str, length num_models
    - metric: str, metric by which the scores were calculated
    '''
    assert len(scores) == len(legends)
    num_models, nchunks = scores.shape

    xticklabels = []
    start = 0
    for i in range(nchunks):
        end = int(np.ceil(100.0 / nchunks * (i + 1)))
        label = f'{start:d}-{end:d}%'
        xticklabels.append(label)
        start = end

    df = pd.DataFrame(data=scores.T, columns=legends, index=xticklabels)
    fig, ax = plt.subplots(1, 1, figsize=[8, 6], constrained_layout=True)
    df.plot(kind='bar', ax=ax, width=0.8)

    # rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', rotation_mode='anchor')
    ax.set_xlabel('chunk of data')
    ax.set_ylabel(metric)
    ax.set_title('Model Performance vs. % chunk of data')
    ax.grid()
    plt.show()
