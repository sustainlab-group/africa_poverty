from collections import defaultdict
from glob import glob
import os
import time
from typing import Any, DefaultDict, Dict, Mapping

import numpy as np
import tensorflow as tf


def param_to_str(p: float) -> str:
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


def get_full_experiment_name(experiment_name: str, batch_size: int,
                             fc_reg: float, conv_reg: float,
                             learning_rate: float, tag: str = None) -> str:
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


def make_log_and_ckpt_dirs(log_dir_base: str, ckpt_dir_base: str, full_experiment_name: str):
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


def checkpoint_path_exists(ckpt_path: str) -> bool:
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


def load(sess: tf.Session, saver: tf.train.Saver, checkpoint_dir: str) -> bool:
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


def print_number_of_parameters(verbose: bool = True):
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


def run_batches(sess: tf.Session, tensors_dict_ops: Mapping[str, tf.Tensor],
                max_nbatches=None, verbose=False) -> Dict[str, np.ndarray]:
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
    all_tensors = defaultdict(list)  # type: DefaultDict[str, Any]
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


def run_epoch(sess: tf.Session, tensors_dict_ops: Mapping[str, tf.Tensor],
              verbose: bool = False) -> Dict[str, np.ndarray]:
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


def save_results(dir_path: str, np_dict: dict, filename: str = 'features.npz'):
    '''Saves a compressed features.npz file in the given dir.

    Args
    - dir_path: str, path to directory to save .npz file
    - np_dict: dict, maps str => np.array
    - filename: str, name of file to save
    '''
    if not os.path.exists(dir_path):
        print('Creating directory at:', dir_path)
        os.makedirs(dir_path)
    npz_path = os.path.join(dir_path, filename)
    assert not os.path.exists(npz_path), f'Path {npz_path} already existed!'
    for key, nparr in np_dict.items():
        print(f'{key}: shape {nparr.shape}, dtype {nparr.dtype}')
    print(f'Saving results to {npz_path}')
    np.savez_compressed(npz_path, **np_dict)


def check_existing(models: dict, logs_root_dir: str, ckpts_root_dir: str,
                   save_filename: str) -> bool:
    '''
    Args
    - models: dict, models[model_name]['model_dir'] is name of model directory
    - logs_root_dir: str, path to root directory for saving logs
    - ckpts_root_dir: str, path to root directory for saving checkpoints
    - save_filename: str, name of existing file to check for

    Returns: bool, True if ckpts exist and no *.npz files found, otherwise False
    '''
    models_with_results = []
    for model_name in models:
        model_dir = models[model_name]['model_dir']

        # check that checkpoint exists
        ckpt_glob = os.path.join(ckpts_root_dir, model_dir, 'ckpt-*')
        assert len(glob(ckpt_glob)) > 0, f'did not find checkpoint matching: {ckpt_glob}'

        npz_path = os.path.join(logs_root_dir, model_dir, save_filename)
        if os.path.exists(npz_path):
            models_with_results.append(model_dir)

    if len(models_with_results) > 0:
        print('The following model directories contain *.npz files that would be overwritten:')
        print('\n'.join(models_with_results))
        return False
    return True


def run_extraction_on_models(model_infos, ModelClass, model_params, batcher,
                             batches_per_epoch, logs_root_dir, ckpts_root_dir,
                             save_filename, batch_keys=(), feed_dict=None):
    '''
    Args
    - model_infos: list of dict
        - 'model_dir': str, name of folder where model is saved
        - 'bands': tuple
    - ModelClass: class, an instance `model` of ModelClass has attributes
        model.features_layer: tf.Tensor
        model.outputs: tf.Tensor
    - model_params: dict, parameters to pass to ModelClass constructor
    - batcher: Batcher, whose batch_op includes 'images' key
    - batches_per_epoch: int
    - logs_root_dir: str, path to root directory for saving logs
    - ckpts_root_dir: str, path to root directory for saving checkpoints
    - save_filename: str, name of file to save
    - batch_keys: list of str
    - feed_dict: dict, tf.Tensor => python value, feed_dict for initializing batcher iterator
    '''
    print('Building model...')
    init_iter, batch_op = batcher.get_batch()
    model = ModelClass(batch_op['images'], **model_params)
    tensors_dict_ops = {
        'features': model.features_layer,
        'preds': tf.squeeze(model.outputs)
    }
    for key in batch_keys:
        if key in batch_op:
            tensors_dict_ops[key] = batch_op[key]

    saver = tf.train.Saver(var_list=None)
    var_init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    print('Creating session...')
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Session(config=config_proto) as sess:
        sess.run(init_iter, feed_dict=feed_dict)

        for model_info in model_infos:
            model_dir = model_info['model_dir']
            ckpt_dir = os.path.join(ckpts_root_dir, model_dir)
            logs_dir = os.path.join(logs_root_dir, model_dir)

            # clear the model weights, then load saved checkpoint
            print('Loading saved ckpt...')
            sess.run(var_init_ops)
            load(sess, saver, ckpt_dir)

            # run the saved model, then save to *.npz files
            all_tensors = run_batches(
                sess, tensors_dict_ops, max_nbatches=batches_per_epoch, verbose=True)
            save_results(dir_path=logs_dir, np_dict=all_tensors, filename=save_filename)
