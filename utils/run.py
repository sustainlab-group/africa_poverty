from utils.analysis import evaluate

from collections import defaultdict
import os
import time

import numpy as np
import PIL.Image
import PIL.ImageDraw
import tensorflow as tf


def param_to_str(p):
    '''
    if p < 1, only leaves everything after decimal
    - 0.001 -> '001'
    - 1e-06 -> '1e-06'

    if p >= 1, adds a decimal point if there isn't already one
    - 1 -> '1.'
    - 2.5 -> '2.5'
    '''
    if p < 1:
        return str(p).split('.')[-1]
    elif '.' in str(p):
        return str(p)
    else:
        return str(p) + '.'


def get_full_experiment_name(experiment_name, batch_size, fc_reg, conv_reg, learning_rate, tag=None):
    '''Returns a str '{experiment_name}_b{batch_size}_fc{fc_str}_conv{conv_str}' where fc_str and
    conv_str are the numbers past the decimal for the fc/conv regularization parameters. Optionally
    appends a tag to the end.

    Args
    - experiment_name: str
    - batch_size: int
    - fc_reg: float
    - conv_reg: float
    - learning_rate: float
    - tag: str or None

    Returns
    - full_experiment_name: str
    '''
    fc_str = param_to_str(fc_reg)
    conv_str = param_to_str(conv_reg)
    lr_str = param_to_str(learning_rate)
    full_experiment_name = f'{experiment_name}_b{batch_size}_fc{fc_str}_conv{conv_str}_lr{lr_str}'

    if tag is not None:
        full_experiment_name += f'_{tag}'

    return full_experiment_name


def make_log_and_ckpt_dirs(log_dir_base, ckpt_dir_base, full_experiment_name):
    '''Creates 2 new directories:
      1. log_dir: log_dir_base/full_experiment_name
      2. ckpt_dir: ckpt_dir_base/full_experiment_name

    Args
    - log_dir_base: str, path to base directory for logs from all experiments
    - ckpt_dir_base: str, path to base directory for checkpoints from all experiments
    - full_experiment_name: str

    Returns: log_dir, ckpt_dir
    - log_dir: str, path to log directory for this specific experiment
    - ckpt_prefix: str, prefix for checkpoint files for this experiment, equals ckpt_dir/ckpt
    '''
    log_dir = os.path.join(log_dir_base, full_experiment_name)
    ckpt_dir = os.path.join(ckpt_dir_base, full_experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_prefix = os.path.join(ckpt_dir, 'ckpt')
    return log_dir, ckpt_prefix


def checkpoint_path_exists(ckpt_path):
    '''
    Args
    - ckpt_path: str, path to a checkpoint file

    Returns: bool, whether the checkpoint path exists
    '''
    if ckpt_path[-6:] == '.index':
        ckpt_path = ckpt_path[-6:]
    if ckpt_path[-5:] == '.meta':
        ckpt_path = ckpt_path[-5:]
    return os.path.exists(ckpt_path + '.index') or os.path.exists(ckpt_path + '.meta')


class LoadNoFileError(Exception):
    pass


def load(sess, saver, checkpoint_dir):
    '''Loads the most recent checkpoint from checkpoint_dir.

    Args
    - sess: tf.Session
    - saver: tf.train.Saver
    - checkpoint_dir: str, path to directory containing checkpoint(s)

    Returns: bool, True if successful at restoring checkpoint from given directory
    '''
    print(f'Reading from checkpoint dir: {checkpoint_dir}')
    if checkpoint_dir is None:
        raise ValueError('No checkpoint path, given, cannot load checkpoint')
    if not os.path.isdir(checkpoint_dir):
        raise ValueError('Given path is not a valid directory.')

    # read the CheckpointState proto from the 'checkpoint' file in checkpoint_dir
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(f'Loading checkpoint: {ckpt_name}')
        if not checkpoint_path_exists(ckpt.model_checkpoint_path):
            raise LoadNoFileError('Checkpoint could not be loaded because it does not exist,'
                                  ' but its information is in the checkpoint meta-data file.')
        saver.restore(sess, ckpt.model_checkpoint_path)
        return True
    return False


def print_number_of_parameters(verbose=True):
    '''Prints the total number of trainable parameters.

    Args
    - verbose: bool, whether to print name and shape info for every trainable var
    '''
    total_parameters = 0  # total # of trainable params in the current graph
    num_none_vars = 0  # num variables in the graph with a shape that is not fully defined

    for variable in tf.trainable_variables():
        name = variable.name
        shape = (d.value for d in variable.shape)  # each d is a tf.Dimension
        num_params = np.prod(variable.shape).value

        if verbose:
            print(f'Variable name: {name}, shape: {shape}, num_params: {num_params}')

        if num_params is None:
            num_none_vars += 1
        else:
            total_parameters += num_params

    print('Total parameters:', total_parameters)


def evaluate_preds(sess, epoch, split, labels, preds, summary_writer=None, eval_summaries=None):
    '''
    Args
    - sess: tf.Session
    - epoch: int
    - split: str, one of ['train', 'val', 'test']
    - labels: np.array, shape [N]
    - preds: np.array, shape [N]
    - summary_writer: tf.SummaryWriter
    - eval_summaries: dict, keys are str
        'r2_placeholder' => tf.placeholder
        'R2_placeholder' => tf.placeholder
        'mse_placeholder' => tf.placeholder
        'summary_op' => tf.Summary, merge of r2, R2, and mse summaries

    Returns
    - r2: float, squared Pearson correlation coefficient between preds and labels
    - R2: float, R^2 (coefficient of determination) between preds and labels
    - mse: float, mean-squared-error between preds and labels
    - rank: float, Spearman rank correlation coefficient between preds and labels
    '''
    assert len(preds) == len(labels)
    num_examples = len(labels)

    r2, R2, mse, rank = evaluate(labels, preds, do_print=False)
    s = 'Epoch {:02d}, {} ({} examples) r^2: {:0.3f}, R^2: {:0.3f}, mse: {:0.3f}, rank: {:0.3f}'
    print(s.format(epoch, split, num_examples, r2, R2, mse, rank))

    if summary_writer is not None and eval_summaries is not None:
        summary_str = sess.run(eval_summaries['summary_op'], feed_dict={
            eval_summaries['r2_placeholder']: r2,
            eval_summaries['R2_placeholder']: R2,
            eval_summaries['mse_placeholder']: mse
        })
        summary_writer.add_summary(summary_str, epoch)
    return r2, R2, mse


def run_batches(sess, tensors_dict_ops, max_nbatches=None, verbose=False):
    '''Runs the ops in tensors_dict_ops for a fixed number of batches or until
    reaching a tf.errors.OutOfRangeError, concatenating the runs.

    Note: assumes that the dataset iterator doesn't need initialization, or is
        already initialized.

    Args
    - sess: tf.Session
    - tensors_dict_ops: dict, str => tf.Tensor, shape [batch_size] or [batch_size, D]
    - max_nbatches: int, maximum number of batches to run the ops for,
        set to None to run until reaching a tf.errors.OutOfRangeError
    - verbose: bool, whether to print out current batch and speed

    Returns
    - all_tensors: dict, str => np.array, shape [N] or [N, D]
    '''
    all_tensors = defaultdict(list)
    curr_batch = 0
    start_time = time.time()
    try:
        while True:
            tensors_dict = sess.run(tensors_dict_ops)
            for name, arr in tensors_dict.items():
                all_tensors[name].append(arr)
            curr_batch += 1
            if verbose:
                speed = curr_batch / (time.time() - start_time)
                print(f'\rRan {curr_batch} batches ({speed:.3f} batch/s)', end='')
            if (max_nbatches is not None) and (curr_batch >= max_nbatches):
                break
    except tf.errors.OutOfRangeError:
        pass

    print()  # print a newline, since the previous print()'s don't print newlines
    for name in all_tensors:
        all_tensors[name] = np.concatenate(all_tensors[name])
    return all_tensors


def run_epoch(sess, tensors_dict_ops, verbose=False):
    '''
    Args
    - sess: tf.Session
    - tensors_dict_ops: dict, str => tf.Tensor, shape [batch_size] or [batch_size, D]
    - verbose: bool, whether to print out current batch and speed

    Note: assumes that the dataset iterator doesn't need initialization, or is
        already initialized.

    Returns
    - all_tensors: dict, str => np.array, shape [N] or [N, D]
    '''
    return run_batches(sess=sess, tensors_dict_ops=tensors_dict_ops, verbose=verbose)


def create_eval_summaries(scope):
    '''
    Args
    - scope: str

    Returns metrics: dict, keys are str
        'r2_placeholder' => tf.placeholder
        'R2_placeholder' => tf.placeholder
        'mse_placeholder' => tf.placeholder
        'summary_op' => tf.Summary, merge of r2, R2, and mse summaries
    '''
    metrics = {}
    # not sure why, but we need the '/' in order to reuse the same 'train/' name for the scope
    with tf.name_scope(scope + '/'):
        metrics['r2_placeholder'] = tf.placeholder(tf.float32, shape=[], name='r2_placeholder')
        metrics['R2_placeholder'] = tf.placeholder(tf.float32, shape=[], name='R2_placeholder')
        metrics['mse_placeholder'] = tf.placeholder(tf.float32, shape=[], name='mse_placeholder')
        metrics['summary_op'] = tf.summary.merge([
            tf.summary.scalar('r2', metrics['r2_placeholder']),
            tf.summary.scalar('R2', metrics['R2_placeholder']),
            tf.summary.scalar('mse', metrics['mse_placeholder'])
        ])
    return metrics


def add_image_summaries(images, labels, preds, locs, k=1):
    '''Adds image summaries for the k best and k worst images in each batch.
    Each image is overlayed with (lat, lon), label, and prediction.

    Args
    - images: tf.Tensor, shape [batch_size, H, W, C], type float32
        - C must be either 3 (RGB order), or 1 (grayscale)
        - must already be standardized (relative to entire dataset) with mean 0, std 1
    - labels: tf.Tensor, shape [batch_size]
    - preds: tf.Tensor, shape [batch_size]
    - locs: tf.Tensor, shape [batch_size, 2], each row is [lat, lon]
    - k: int, number of best and worst images to show per batch

    Returns
    - summaries: tf.summary, merged summaries
    '''
    # For float tensors, tf.summary.image automatically scales min/max to 0/255.
    # Set +/- 3 std. dev. to 0/255.
    # We want to display images with our own scaling -> cast to tf.uint8
    images = tf.clip_by_value((images / 6.0 + 0.5) * 255, clip_value_min=0, clip_value_max=255)
    images = tf.cast(images, tf.uint8)

    def write_on_imgs(imgs, locs, labels, preds):
        '''Writes white text w/ black background onto images

        Args
        - imgs: np.array, shape [num_imgs, H, W, C], type uint8
            C must be either 1 or 3
        - locs: np.array, shape [num_imgs, 2]
        - labels: np.array, shape [num_imgs]
        - preds: np.array, shape [num_imgs]

        Returns
        - new_imgs: np.array, shape [num_imgs, H, W, C]
        '''
        C = imgs.shape[3]
        new_imgs = np.empty_like(imgs)
        for i, img in enumerate(imgs):
            if C == 1:
                img = img[:, :, 0]  # remove C dim. new shape: [H, W]
            img = PIL.Image.fromarray(img)
            # write white text on black background
            draw = PIL.ImageDraw.Draw(img)
            text = 'loc: ({:.6f}, {:.6f})\nlabel: {:.4f}, pred: {:.4f}'.format(
                locs[i][0], locs[i][1], labels[i], preds[i])
            size = draw.textsize(text)  # (w, h) of text
            draw.rectangle(xy=[(0, 0), size], fill='black')
            draw.text(xy=(0, 0), text=text, fill='white')
            if C == 1:
                new_imgs[i, :, :, 0] = np.asarray(img)
            else:
                new_imgs[i] = np.asarray(img)
        return new_imgs

    diff = tf.abs(preds - labels)
    _, worst_indices = tf.nn.top_k(diff, k=k)
    _, best_indices = tf.nn.top_k(-1 * diff, k=k)
    worst_inputs = [tf.gather(x, worst_indices) for x in [images, locs, labels, preds]]
    worst_img_sum = tf.summary.image(
        'worst_images_in_batch',
        tf.py_func(func=write_on_imgs, inp=worst_inputs, Tout=tf.uint8, stateful=False, name='write_on_worst_imgs'),
        max_outputs=k)
    best_inputs = [tf.gather(x, best_indices) for x in [images, locs, labels, preds]]
    best_img_sum = tf.summary.image(
        'best_images_in_batch',
        tf.py_func(func=write_on_imgs, inp=best_inputs, Tout=tf.uint8, stateful=False, name='write_on_best_imgs'),
        max_outputs=k)

    return tf.summary.merge([worst_img_sum, best_img_sum])
