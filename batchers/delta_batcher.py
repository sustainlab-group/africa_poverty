from batchers.batcher import Batcher, get_lsms_tfrecord_paths
from batchers.dataset_constants import SURVEY_NAMES

from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf


def get_lsms_tfrecord_pairs(indices_dict):
    '''
    Args
    - indices_dict: maps str => list of indices
        or None to get all pairs

    Returns: np.array or dict
    - if indices_dict is None, returns:
        - pairs: np.array, shape [N, 2], type str
    - otherwise, returns:
        - paths_dict: maps str => np.array, shape [N, 2], type str
            each row is [path1, path2], corresponds to TFRecords containing
            images of the same location such that year1 < year2
    '''
    tfrecord_paths = np.asarray(get_lsms_tfrecord_paths(SURVEY_NAMES['LSMS']))
    delta_pairs_df = pd.read_csv('/atlas/u/chrisyeh/africa_poverty/data/lsms_deltas_pairs.csv')

    if indices_dict is None:
        paths1 = tfrecord_paths[delta_pairs_df['index1'].values]
        paths2 = tfrecord_paths[delta_pairs_df['index2'].values]
        pairs = np.stack([paths1, paths2], axis=1)
        return pairs
    else:
        paths_dict = defaultdict(list)
        for i1, i2 in zip(delta_pairs_df['index1'], delta_pairs_df['index2']):
            for k, indices in indices_dict.items():
                isin = np.isin([i1, i2], indices)
                assert isin[0] == isin[1]
                if isin[0]:
                    pair = tfrecord_paths[[i1, i2]]
                    paths_dict[k].append(pair)
        for k in paths_dict:
            paths_dict[k] = np.stack(paths_dict[k], axis=0)
        return paths_dict


class DeltaBatcher(Batcher):
    def __init__(self, tfrecord_pairs, dataset, batch_size, label_name,
                 num_threads=1, epochs=1, ls_bands='rgb', nl_band=None, orig_labels=False,
                 shuffle=True, augment=True, negatives='zero', normalize=True, cache=False):
        '''
        Args
        - tfrecord_pairs: tf.Tensor, type str, shape [N, 2], each row is [path1, path2]
            - pairs of paths to TFRecord files containing satellite images
        - orig_labels: bool, whether to include the original labels (for multi-task training)
        - see Batcher class for other args
        - does not allow label_name to be None
        - does not allow for nl_label
        '''
        assert label_name is not None
        self.orig_labels = orig_labels

        super(DeltaBatcher, self).__init__(
            tfrecord_files=tfrecord_pairs,
            dataset=dataset,
            batch_size=batch_size,
            label_name=label_name,
            num_threads=num_threads,
            epochs=epochs,
            ls_bands=ls_bands,
            nl_band=nl_band,
            nl_label=None,
            shuffle=shuffle,
            augment=augment,
            negatives=negatives,
            normalize=normalize,
            cache=cache)

    def get_batch(self):
        '''Gets the tf.Tensors that represent a batch of data.

        Returns
        - iter_init: tf.Operation that should be run before each epoch
        - batch: dict, str -> tf.Tensor
            - 'images': tf.Tensor, shape [batch_size, H, W, C], type float32
                - C depends on the ls_bands and nl_band settings
            - 'locs': tf.Tensor, shape [batch_size, 2], type float32, each row is [lat, lon]
            - 'labels': tf.Tensor, shape [batch_size] or [batch_size, 3], type float32
                - shape [batch_size, 3] if self.orig_labels = True
            - 'years1': tf.Tensor, shape [batch_size], type int32
            - 'years2': tf.Tensor, shape [batch_size], type int32

        IMPLEMENTATION NOTE: The order of tf.data.Dataset.batch() and .repeat() matters!
            Suppose the size of the dataset is not evenly divisible by self.batch_size.
            If batch then repeat, ie. `ds.batch(batch_size).repeat(num_epochs)`:
                the last batch of every epoch will be smaller than batch_size
            If repeat then batch, ie. `ds.repeat(num_epochs).batch(batch_size)`:
                the boundaries between epochs are blurred, ie. the dataset "wraps around"
        '''
        datasets = []
        for idx in [0, 1]:
            ds = tf.data.TFRecordDataset(
                filenames=self.tfrecord_files[:, idx],
                compression_type='GZIP',
                buffer_size=1024 * 1024 * 128,  # 128 MB buffer size
                num_parallel_reads=int(self.num_threads / 2))

            ds = ds.map(self.process_tfrecords, num_parallel_calls=self.num_threads)
            if self.nl_band == 'split':
                ds = ds.map(self.split_nl_band)

            datasets.append(ds)

        ds = tf.data.Dataset.zip(tuple(datasets))
        ds = ds.map(self.merge_examples)

        if self.cache:
            ds = ds.cache()
        if self.shuffle:
            ds = ds.shuffle(buffer_size=1000)
        if self.augment:
            ds = ds.map(self.augment_example)

        # batch then repeat => batches respect epoch boundaries
        # - i.e. last batch of each epoch might be smaller than batch_size
        ds = ds.batch(self.batch_size)
        ds = ds.repeat(self.epochs)

        # prefetch 2 batches at a time
        ds = ds.prefetch(2)

        iterator = ds.make_initializable_iterator()
        batch = iterator.get_next()
        iter_init = iterator.initializer
        return iter_init, batch

    def merge_examples(self, ex1, ex2):
        '''
        Args
        - ex1, ex2: each exN is a dict
            - 'images': tf.Tensor, shape [224, 224, C], type float32
                - channel order is [B, G, R, SWIR1, SWIR2, TEMP1, NIR, DMSP, VIIRS]
            - 'labels': tf.Tensor, scalar, type float32
                - default value of np.nan if self.label_name is not a key in the protobuf
            - 'locs': tf.Tensor, type float32, shape [2], order is [lat, lon]
            - 'years': tf.Tensor, scalar, type int32
                - default value of -1 if 'year' is not a key in the protobuf

        Returns: merged, dict
        - 'images': tf.Tensor, shape [224, 224, C], type float32
            - channel order is [B, G, R, SWIR1, SWIR2, TEMP1, NIR, DMSP, VIIRS]
        - 'labels': tf.Tensor, shape scalar or [3], type float32
            - shape [3] if self.orig_labels = True
        - 'locs': tf.Tensor, shape [2], type float32, order is [lat, lon]
        - 'years1': tf.Tensor, scalar, type int32
        - 'years2': tf.Tensor, scalar, type int32
        '''
        assert_op = tf.assert_equal(ex1['locs'], ex2['locs'])
        with tf.control_dependencies([assert_op]):
            merged = {
                'images': tf.concat([ex1['images'], ex2['images']], axis=2),
                'labels': ex2['labels'] - ex1['labels'],
                'locs': ex1['locs'],
                'years1': ex1['years'],
                'years2': ex2['years'],
            }
        if self.orig_labels:
            merged['labels'] = tf.stack([merged['labels'], ex1['labels'], ex2['labels']])
        return merged

    def augment_example(self, ex):
        '''Performs image augmentation: random flips + using a single image.

        Args
        - ex: dict {'images': img, 'labels': label, ...}
            - img: tf.Tensor, shape [H, W, 2*C], type float32
                - if self.nl_band is not None, final band is NL
            - label: tf.Tensor, scalar or shape [3], type float32
                - shape [3] if self.orig_labels = True

        Returns: ex, with img replaced with an augmented image
        '''
        assert self.augment
        img = ex['images']
        label = ex['labels']

        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_flip_left_right(img)
        C = int(int(img.shape[2]) / 2)

        # uniform var in [0, 1)
        p = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if self.orig_labels:
            delta, label1, label2 = ex['labels'][0], ex['labels'][1], ex['labels'][2]
            img1 = img[:, :, :C]
            img2 = img[:, :, C:]
            ex['images'], ex['labels'] = tf.case(
                {
                    p < 1/8:  # prob 1/8, use 1st image only
                        lambda: (tf.concat([img1, img1], axis=2),
                                 tf.stack([0.0, label1, label1])),
                    (p >= 1/8) & (p < 2/8):  # prob 1/8, use 2nd image only
                        lambda: (tf.concat([img2, img2], axis=2),
                                 tf.stack([0.0, label2, label2])),
                    (p >= 2/8) & (p < 5/8):  # prob 3/8, flip image order
                        lambda: (tf.concat([img2, img1], axis=2),
                                 tf.stack([-delta, label2, label1]))
                },
                default=lambda: (img, label))  # prob 3/8, do nothing
        else:
            img1 = img[:, :, :C]
            img2 = img[:, :, C:]
            ex['images'], ex['labels'] = tf.case(
                {
                    p < 1/8:  # prob 1/8, use 1st image only
                        lambda: (tf.concat([img1, img1], axis=2), 0.0),
                    (p >= 1/8) & (p < 2/8):  # prob 1/8, use 2nd image only
                        lambda: (tf.concat([img2, img2], axis=2), 0.0),
                    (p >= 2/8) & (p < 5/8):  # prob 3/8, flip image order
                        lambda: (tf.concat([img2, img1], axis=2), -label)
                },
                default=lambda: (img, label))  # prob 3/8, do nothing
        return ex


class DeltaClassBatcher(DeltaBatcher):
    def __init__(self, tfrecord_pairs, dataset, batch_size, label_name,
                 num_threads=1, epochs=1, ls_bands='rgb', nl_band=None,
                 shuffle=True, augment=True, negatives='zero', normalize=True, cache=False):
        '''
        Args
        - see DeltaBatcher class for other args
        - does not allow orig_labels
        '''
        assert label_name is not None

        super(DeltaClassBatcher, self).__init__(
            tfrecord_pairs=tfrecord_pairs,
            dataset=dataset,
            batch_size=batch_size,
            label_name=label_name,
            num_threads=num_threads,
            epochs=epochs,
            ls_bands=ls_bands,
            nl_band=nl_band,
            orig_labels=False,
            shuffle=shuffle,
            augment=augment,
            negatives=negatives,
            normalize=normalize,
            cache=cache)

    def merge_examples(self, ex1, ex2):
        '''
        Args
        - ex1, ex2: each exN is a dict
            - 'images': tf.Tensor, shape [224, 224, C], type float32
                - channel order is [B, G, R, SWIR1, SWIR2, TEMP1, NIR, DMSP, VIIRS]
            - 'labels': tf.Tensor, scalar, type float32
                - default value of np.nan if self.label_name is not a key in the protobuf
            - 'locs': tf.Tensor, type float32, shape [2], order is [lat, lon]
            - 'years': tf.Tensor, scalar, type int32
                - default value of -1 if 'year' is not a key in the protobuf

        Returns: merged, dict
        - 'images': tf.Tensor, shape [224, 224, C], type float32
            - channel order is [B, G, R, SWIR1, SWIR2, TEMP1, NIR, DMSP, VIIRS]
        - 'labels': tf.Tensor, shape scalar, type int32
        - 'locs': tf.Tensor, shape [2], type float32, order is [lat, lon]
        - 'years1': tf.Tensor, scalar, type int32
        - 'years2': tf.Tensor, scalar, type int32
        '''
        assert_op = tf.assert_equal(ex1['locs'], ex2['locs'])
        with tf.control_dependencies([assert_op]):
            d = ex2['labels'] - ex1['labels']
            delta_class = tf.case({
                d < -0.125: lambda: 0,
                (d >= -0.125) & (d <= 0.125): lambda: 1,
                d > 0.125: lambda: 2
            })
            merged = {
                'images': tf.concat([ex1['images'], ex2['images']], axis=2),
                'labels': delta_class,
                'locs': ex1['locs'],
                'years1': ex1['years'],
                'years2': ex2['years'],
            }
        return merged

    def augment_example(self, ex):
        '''Performs image augmentation: random flips + using a single image.

        Args
        - ex: dict {'images': img, 'labels': label, ...}
            - img: tf.Tensor, shape [H, W, 2*C], type float32
                - if self.nl_band is not None, final band is NL
            - label: tf.Tensor, scalar, type int32
                - 0 = neg. delta, 1 = no delta, 2 = pos. delta

        Returns: ex, with img replaced with an augmented image
        '''
        assert self.augment
        img = ex['images']
        label = ex['labels']

        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_flip_left_right(img)
        C = int(int(img.shape[2]) / 2)

        # uniform var in [0, 1)
        p = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        ex['images'], ex['labels'] = tf.case(
            {
                p < 1/8:  # prob 1/8, use 1st image only
                    lambda: (tf.concat([img[:, :, :C], img[:, :, :C]], axis=2), 1),
                tf.logical_and(p >= 1/8, p < 2/8):  # prob 1/8, use 2nd image only
                    lambda: (tf.concat([img[:, :, C:], img[:, :, C:]], axis=2), 1),
                tf.logical_and(p >= 2/8, p < 5/8):  # prob 3/8, flip image order
                    lambda: (tf.concat([img[:, :, C:], img[:, :, :C]], axis=2), -label + 2)
            },
            default=lambda: (img, label))  # prob 3/8, do nothing
        return ex
