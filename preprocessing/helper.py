import sys
import time
from typing import Dict, List, Mapping, Tuple

import numpy as np
import tensorflow as tf

sys.path.append('../')
from utils.general import add_to_heap

def parse_record_str(record_str: str):
    '''Parses a record str and returns the feature map.

    Args
    - record_str: str, binary representation of Example message
    '''
    # parse binary string into Example message
    ex = tf.train.Example.FromString(record_str)
    features = ex.features  # get Features message within the Example
    feature_map = features.feature  # get mapping from feature name strings to Feature
    return feature_map


def get_first_feature_map(tfrecord_path: str):
    '''Gets feature_map dict of 1st TFRecord in a TFRecord file.

    Args
    - tfrecord_path: str, path to a TFRecord file with GZIP compression

    Returns
    - feature_map: protobuf map from feature name strings to Feature
    '''
    # Create an iterator over the TFRecords file. The iterator yields
    # the binary representations of Example messages as strings.
    options = tf.io.TFRecordOptions(tf.io.TFRecordCompressionType.GZIP)
    iterator = tf.io.tf_record_iterator(tfrecord_path, options=options)

    # get the first Example stored in the TFRecords file
    record_str = next(iterator)
    feature_map = parse_record_str(record_str)
    return feature_map


def get_feature_types(feature_map):
    '''Gets the types and shapes of each feature in a given feature_map.

    Args
    - feature_map: protobuf map from feature name strings to Feature

    Returns
    - feature_types: dict, maps feature names (str) to tuple of (ft_type, ft_shape)
    '''
    # use the WhichOneof() method on messages with `oneof` fields to
    # determine the type of the field
    feature_types = {}
    for name in feature_map.keys():
        ft_type = feature_map[name].WhichOneof('kind')
        ft_shape = np.array(feature_map[name].__getattribute__(ft_type).value).shape
        feature_types[name] = (ft_type, ft_shape)
    return feature_types


def print_scalar_values(feature_map):
    '''Prints scalar values from a TFRecord feature map.

    Args
    - feature_map: protobuf map from feature name strings to Feature
    '''
    for name in sorted(feature_map.keys()):
        ft_type = feature_map[name].WhichOneof('kind')
        ft_shape = np.array(feature_map[name].__getattribute__(ft_type).value).shape
        if ft_type == 'float_list' and ft_shape == (1,):
            value = feature_map[name].float_list.value[0]
            print(f'{name}: {value}')
        elif ft_type == 'bytes_list' and ft_shape == (1,):
            value = feature_map[name].bytes_list.value[0].decode()
            print(f'{name}: {value}')


def analyze_tfrecord_batch(iter_init, batch_op, total_num_images, nbands, k=20):
    '''
    Args
    - iter_init: tf.Op, operation to initialize iterator, or None if not needed
    - batch_op: dict, str -> tf.Tensor
        - 'images': tf.Tensor, type float32, shape [batch_size, 224, 224, nbands]
        - 'locs': tf.Tensor, type float32, shape [batch_size, 2], each row is [lat, lon]
        - 'labels': tf.Tensor, type float32, shape [batch_size]
        - 'years': tf.Tensor, type int32, shape [batch_size]
    - total_num_images: int
    - nbands: int
    - k: int, number of worst images to track

    Returns: stats, k_worst
    - stats: dict
        - 'num_good_pixels': np.array, shape [images_count], type int, number of good pixels per image
        - 'mins_nz': np.array, shape [nbands], type float64, min value per band excluding 0s
        - 'maxs': np.array, shape [nbands], type float64, max value per band
        - 'sums': np.array, shape [nbands], type float64, sum of values per band
        - 'sum_sqs': np.array, shape [nbands], type float64, sum of squared-values per band
        - 'nz_pixels': np.array, shape [nbands], type int64, number of non-zero pixels per band
    - k_worst: list of length k, elements are (value, (label, image, loc, year))
        - value = -number of good pixels
        - tracks the top-k worst images (in terms of # of good pixels)
    '''
    images_count = 0

    # statistics for each band: min, max, sum, sum of squares, number of non-zero pixels
    mins = np.ones(nbands, dtype=np.float64) * np.inf
    mins_nz = np.ones(nbands, dtype=np.float64) * np.inf
    mins_goodpx = np.ones(nbands, dtype=np.float64) * np.inf
    maxs = np.zeros(nbands, dtype=np.float64)
    sums = np.zeros(nbands, dtype=np.float64)
    sum_sqs = np.zeros(nbands, dtype=np.float64)
    nz_pixels = np.zeros(nbands, dtype=np.int64)

    # heap to track the worst (by -nz_pixels) images
    # - elements are (value, (label, image, loc))
    k_worst = []

    batch_times = []
    processing_times = []
    start = time.time()

    # number of `good pixels` in each image
    num_good_pixels = []

    with tf.Session() as sess:
        if iter_init is not None:
            sess.run(iter_init)

        while True:
            try:
                batch_start_time = time.time()
                batch_np = sess.run(batch_op)
                img_batch, loc_batch, label_batch, year_batch = \
                    batch_np['images'], batch_np['locs'], batch_np['labels'], batch_np['years']
                batch_size = len(img_batch)

                processing_start_time = time.time()
                batch_times.append(processing_start_time - batch_start_time)

                dmsp_mask = (year_batch < 2012)
                dmsp_bands = np.arange(nbands-1)
                viirs_mask = ~dmsp_mask
                viirs_bands = [i for i in range(nbands) if i != nbands-2]

                # a good pixel is one where at least 1 band is > 0
                batch_goodpx = np.any(img_batch > 0, axis=3)
                num_good_pixels_per_image = np.sum(batch_goodpx, axis=(1,2))
                num_good_pixels.extend(num_good_pixels_per_image)

                img_batch_positive = np.where(img_batch <= 0, np.inf, img_batch)
                img_batch_nonneg = np.where(img_batch < 0, 0, img_batch)

                for mask, bands in [(dmsp_mask, dmsp_bands), (viirs_mask, viirs_bands)]:
                    if np.sum(mask) == 0: continue

                    imgs = img_batch[mask]
                    imgs_positive = img_batch_positive[mask]
                    imgs_nonneg = img_batch_nonneg[mask]

                    goodpx = batch_goodpx[mask]
                    imgs_goodpx = imgs[goodpx]  # shape [len(mask), nbands]

                    mins[bands] = np.minimum(mins[bands], np.min(imgs, axis=(0,1,2)))
                    mins_nz[bands] = np.minimum(mins_nz[bands], np.min(imgs_positive, axis=(0,1,2)))
                    mins_goodpx[bands] = np.minimum(mins_goodpx[bands], np.min(imgs_goodpx, axis=0))
                    maxs[bands] = np.maximum(maxs[bands], np.max(imgs, axis=(0,1,2)))

                    # use dtype=np.float64 to avoid significant loss of precision in np.sum
                    sums[bands] += np.sum(imgs_nonneg, axis=(0,1,2), dtype=np.float64)
                    sum_sqs[bands] += np.sum(imgs_nonneg ** 2, axis=(0,1,2), dtype=np.float64)
                    nz_pixels[bands] += np.sum(imgs > 0, axis=(0,1,2))

                # update the k-worst heap
                for i in range(batch_size):
                    data = (label_batch[i], year_batch[i], tuple(loc_batch[i]), img_batch[i])
                    add_to_heap(k_worst, k=k, value=-num_good_pixels_per_image[i], data=data)

                processing_times.append(time.time() - processing_start_time)

                images_count += batch_size
                if images_count % 1024 == 0:
                    print(f'\rProcessed {images_count}/{total_num_images} images...', end='')
            except tf.errors.OutOfRangeError:
                break

    total_time = time.time() - start
    assert len(num_good_pixels) == images_count
    assert images_count == total_num_images

    print(f'\rFinished. Processed {images_count} images.')
    print('Time per batch - mean: {:0.3f}s, std: {:0.3f}s'.format(
        np.mean(batch_times), np.std(batch_times)))
    print('Time to process each batch - mean: {:0.3f}s, std: {:0.3f}s'.format(
        np.mean(processing_times), np.std(processing_times)))
    print('Total time: {:0.3f}s, Num batches: {}'.format(total_time, len(batch_times)))

    stats = {
        'num_good_pixels': num_good_pixels,
        'mins': mins,
        'mins_nz': mins_nz,
        'mins_goodpx': mins_goodpx,
        'maxs': maxs,
        'sums': sums,
        'sum_sqs': sum_sqs,
        'nz_pixels': nz_pixels
    }
    k_worst.sort()
    return stats, k_worst


def print_analysis_results(stats: Mapping[str, np.ndarray],
                           band_order: List[str]) -> Tuple[Dict, Dict]:
    '''
    Args
    - stats: dict
      - 'num_good_pixels': np.array, shape [images_count], type int, number of good pixels per image
      - 'mins': np.array, shape [nbands], type float64, min value per band
      - 'mins_nz': np.array, shape [nbands], type float64, min value per band excluding non-positive
      - 'mins_goodpx': np.array, shape [nbands], type float64, min value per band excluding bad pixels
      - 'maxs': np.array, shape [nbands], type float64, max value per band
      - 'sums': np.array, shape [nbands], type float64, sum of values per band
      - 'sum_sqs': np.array, shape [nbands], type float64, sum of squared-values per band
      - 'nz_pixels': np.array, shape [nbands], type int64, number of non-zero pixels per band
    - band_order: list of str, names of bands

    Returns
    - means: dict, band_name => np.float64, mean of each band excluding bad pixels
    - stds: dict, band_name => np.float64, std. dev. of each band excluding bad pixels
    '''
    num_good_pixels, mins, mins_nz, mins_goodpx, maxs, sums, sum_sqs, nz_pixels = [
        stats[k] for k in
        ['num_good_pixels', 'mins', 'mins_nz', 'mins_goodpx', 'maxs', 'sums', 'sum_sqs', 'nz_pixels']
    ]
    images_count = len(num_good_pixels)
    total_pixels_per_band = images_count * (224 ** 2)  # per band

    print('Statistics including bad pixels')
    means = sums / float(total_pixels_per_band)
    stds = np.sqrt(sum_sqs/float(total_pixels_per_band) - means**2)
    for i, band_name in enumerate(band_order):
        print('Band {:8s} - mean: {:10.6f}, std: {:>9.6f}, min: {:>11.6g}, max: {:11.6f}'.format(
            band_name, means[i], stds[i], mins[i], maxs[i]))

    print('')
    print('Statistics ignoring any 0s and negative values')
    means = sums / nz_pixels
    stds = np.sqrt(sum_sqs/nz_pixels - means**2)
    avg_nz_pixels = nz_pixels.astype(np.float32) / images_count
    for i, band_name in enumerate(band_order):
        print('Band {:8s} - mean: {:10.6f}, std: {:>9.6f}, min: {:>11.6g}, max: {:11.6f}, mean_nz: {:0.6f}'.format(
            band_name, means[i], stds[i], mins_nz[i], maxs[i], avg_nz_pixels[i]))

    print('')
    print('Statistics excluding the bad pixels')
    num_total_pixels = np.sum(num_good_pixels)
    means = sums / float(num_total_pixels)
    stds = np.sqrt(sum_sqs/float(num_total_pixels) - means**2)
    for i, band_name in enumerate(band_order):
        print('Band {:8s} - mean: {:10.6f}, std: {:>9.6f}, min: {:>11.6g}, max: {:11.6f}'.format(
            band_name, means[i], stds[i], mins_goodpx[i], maxs[i]))

    means = {
        band_name: means[b]
        for b, band_name in enumerate(band_order)
    }
    stds = {
        band_name: stds[b]
        for b, band_name in enumerate(band_order)
    }
    return means, stds
