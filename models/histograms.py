import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_per_image_histograms(init_iter, batch_op, band_bin_edges):
    '''
    Args
    - iter_init: tf op, initializes the dataset iterator
    - batch_op: dict, str => tf.Tensor
      - 'images': tf.Tensor, shape [batch_size, 224, 224, C], last channel is nightlights
      - 'labels': tf.Tensor, shape [batch_size]
      - 'locs': tf.Tensor, shape [batch_size, 2]
      - 'years': tf.Tensor, shape [batch_size]
    - band_bin_edges

    Returns: results, dict
    - 'image_hists': np.array, shape [N, C, nbins], type int64
    - 'labels': np.array, shape [N], type float32, all labels
    - 'locs': np.array, shape [N, 2], type float32, all locs
    - 'years': np.array, shape [N], type int32, year for each image
    - 'nls_center': np.array, shape [N], type float32, center nightlight value
    - 'nls_mean': np.array, shape [N], type float32, mean nightlight value
    '''
    keys = ['image_hists', 'labels', 'locs', 'years', 'nls_center', 'nls_mean']
    results = {k: [] for k in keys}

    with tf.Session() as sess:
        sess.run(init_iter)
        try:
            batch_num = 1
            while True:
                batch = sess.run(batch_op)
                results['labels'].append(batch['labels'])
                results['locs'].append(batch['locs'])
                results['years'].append(batch['years'])

                images = batch['images']

                # calculate scalar nightlights
                nl_center = images[:, 111, 111, -1]
                results['nls_center'].append(nl_center)
                nl_mean = np.mean(images[:, :, :, -1], axis=(1, 2))
                results['nls_mean'].append(nl_mean)

                # create image histograms
                num_images = images.shape[0]
                num_bands = images.shape[3]
                for n in range(num_images):
                    image_hists = []
                    image = images[n, :, :, :]
                    for b in range(num_bands):
                        band = image[:, :, b]
                        hist, _ = np.histogram(band, bins=band_bin_edges)
                        image_hists.append(hist)

                    image_hists = np.stack(image_hists)
                    results['image_hists'].append(image_hists)
                print('Finished batch', batch_num)
                batch_num += 1
        except tf.errors.OutOfRangeError:
            pass

    results['image_hists'] = np.stack(results['image_hists'])
    for k in ['labels', 'locs', 'years', 'nls_center', 'nls_mean']:
        results[k] = np.concatenate(results[k])
    return results


def split_nl_hist(image_hists, years):
    '''
    Args
    - image_hists: np.array, shape [nimages, C, nbins], last band is NIGHTLIGHTS
    - years: np.array, shape [nimages]

    Returns
    - image_hists_nl: np.array, shape [nimages, C+1, nbins]
        2nd-to-last band is DMSP, last band is VIIRS
    '''
    nimages, C, nbins = image_hists.shape
    image_hists_nl = np.zeros([nimages, C+1, nbins])
    dmsp_indices = np.where(years < 2012)[0]
    viirs_indices = np.where(years >= 2012)[0]
    dmsp_bands = np.arange(C)  # [0, 1, ..., C-1]
    viirs_bands = np.array(list(range(C-1)) + [C])  # [0, 1, ..., C-2, C]
    for indices, bands in [(dmsp_indices, dmsp_bands), (viirs_indices, viirs_bands)]:
        if np.sum(indices) == 0: continue
        image_hists_nl[indices[:, None], bands, :] = image_hists[indices]
    assert(np.any(image_hists_nl[dmsp_indices, -2:-1, :] > 0))
    assert(np.any(image_hists_nl[viirs_indices, -1:, :] > 0))
    return image_hists_nl


def plot_band_hists(band_hists, band_order, band_colors, bin_centers, xlabel,
                    ylabel, title, yscale='linear'):
    '''
    Args
    - band_hists: np.array, shape [C, nbins]
      - assumes DMSP and VIIRS each have their own band
    - band_order: list of str
    - band_colors: dict, band_name => matplotlib color
    - bin_centers: np.array, shape [nbins]
    - xlabel, ylabel, title: str
    - yscale: str, one of ['linear', 'log']
    '''
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    nbands = band_hists.shape[0]
    for b in range(nbands):
        band_name = band_order[b]
        ax.plot(bin_centers, band_hists[b], label=band_name, color=band_colors[band_name])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, yscale=yscale)
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    plt.show()


def plot_label_hist(labels, bin_edges, title, figsize=(10, 4)):
    '''
    Args
    - labels: np.array, shape [num_images]
    - bin_edges: np.array, shape [num_label_bins + 1]
    - title: str
    - figsize: tuple of (width, height), in inches
    '''
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(labels, bins=bin_edges)
    ax.set(xlabel='label', ylabel='count', title=title)
    ax.grid(True)
    fig.tight_layout()
    plt.show()
