import numpy as np


def load_npz(path, verbose=True):
    '''Loads .npz file into a dict.

    Args
    - path: str, path to .npz file

    Returns
    - result: dict
    '''
    result = {}
    with np.load(path) as npz:
        for key, value in npz.items():
            result[key] = value
            if verbose:
                print('{k}: dtype={d}, shape={s}'.format(k=key, d=value.dtype, s=value.shape))
    return result
