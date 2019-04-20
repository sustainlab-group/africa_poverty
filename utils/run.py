from collections import defaultdict
import os
import time

import numpy as np
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
