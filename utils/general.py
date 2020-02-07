import heapq

import numpy as np


def load_npz(path, verbose=True, check=None):
    '''Loads .npz file into a dict.

    Args
    - path: str, path to .npz file
    - verbose: bool, whether to print out type and shape info
    - check: dict, key (str) => np.array, values to check

    Returns
    - result: dict
    '''
    result = {}
    with np.load(path) as npz:
        for key, value in npz.items():
            result[key] = value
            if verbose:
                print('{k}: dtype={d}, shape={s}'.format(k=key, d=value.dtype, s=value.shape))
    if check is not None:
        for key in check:
            assert key in result
            assert np.allclose(check[key], result[key])
    return result


def colordisplay(df, columns=None, cmap='coolwarm'):
    '''Displays a pandas DataFrame with background color.

    This function should only be called inside a Jupyter Notebook.

    Args
    - df: pd.DataFrame
    - columns: str or list of str, column(s) to color
    - cmap: str, name of matplotlib colormap
    '''
    display(df.style.background_gradient(cmap=cmap, subset=columns))


def add_to_heap(h, k, value, data):
    '''Tracks the max k elements using a heap.

    We will actually use a min-heap for this task. That way, when a new element
    comes in, we compare it to the smallest node in the heap, h[0]. If the new
    value is greater than h[0], we pop h[0] and add the new element in.

    Args
    - h: list, either empty [] or already heapified
    - k: int, desired capacity of the heap
    - value: numeric, value to compare with
    - data: data to store with the value
    '''
    if len(h) < k:
        heapq.heappush(h, (value, data))
    else:
        heapq.heappushpop(h, (value, data))
