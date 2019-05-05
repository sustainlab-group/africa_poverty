from models.resnet_model import Hyperspectral_Resnet
from batchers import delta_batcher
from utils.run import get_full_experiment_name, make_log_and_ckpt_dirs
import utils.trainer

import os
import pickle
from pprint import pprint

import numpy as np
import tensorflow as tf


def run_training(sess, ooc, batcher, dataset, keep_frac, model_name, model_params, batch_size,
                 ls_bands, nl_band, label_name, orig_labels, augment, learning_rate, lr_decay,
                 max_epochs, print_every, eval_every, num_threads, cache, log_dir, save_ckpt_dir,
                 init_ckpt_dir, imagenet_weights_path, hs_weight_init, exclude_final_layer):
    '''
    Args
    - sess: tf.Session
    - ooc: bool, whether to use out-of-country split, must be False
    - batcher: str, batcher type, one of ['delta', 'deltaclass']
    - dataset: str, e.g. 'LSMSDeltaIncountryA'
    - keep_frac: float, only supports 1.0
    - model_name: str, must be 'resnet'
    - model_params: dict
    - batch_size: int
    - ls_bands: one of [None, 'rgb', 'ms']
    - nl_band: one of [None, 'merge', 'split']
    - label_name: str, name of the label in the TFRecord file
    - orig_labels: bool, whether to include original labels for multi-task training
    - augment: bool
    - learning_rate: float
    - lr_decay: float
    - max_epochs: int
    - print_every: int
    - eval_every: int
    - num_threads: int
    - cache: list of str
    - log_dir: str, path to directory to save logs for TensorBoard, must already exist
    - save_ckpt_dir: str, path to checkpoint dir for saving weights
        - intermediate dirs must already exist
    - init_ckpt_dir: str, path to checkpoint dir from which to load existing weights
        - set to empty string '' to use ImageNet or random initialization
    - imagenet_weights_path: str, path to pre-trained weights from ImageNet
        - set to empty string '' to use saved ckpt or random initialization
    - hs_weight_init: str, one of [None, 'random', 'same', 'samescaled']
    - exclude_final_layer: bool, or None
    '''
    # ====================
    #    ERROR CHECKING
    # ====================
    assert os.path.exists(log_dir)
    assert os.path.exists(os.path.dirname(save_ckpt_dir))
    assert keep_frac == 1.0

    if model_name != 'resnet':
        raise NotImplementedError(f'Unsupported model_name: {model_name}')
    if ooc:
        raise NotImplementedError('OOC is currently not supported')
    if 'LSMSDeltaIncountry' not in dataset and 'LSMSDeltaClassIncountry' not in dataset:
        raise NotImplementedError(f'Unsupported dataset: {dataset}')

    if batcher == 'delta':
        Batcher = delta_batcher.DeltaBatcher
        Trainer = utils.trainer.RegressionTrainer
    elif batcher == 'deltaclass':
        assert not orig_labels
        Batcher = delta_batcher.DeltaClassBatcher
        Trainer = utils.trainer.ClassificationTrainer
    else:
        raise NotImplementedError(f'Unsupported batcher: {batcher}')

    model_class = Hyperspectral_Resnet
    # ====================
    #       BATCHERS
    # ====================
    with open('/atlas/u/chrisyeh/africa_poverty/data/lsms_incountry_folds.pkl', 'rb') as f:
        incountry_folds = pickle.load(f)

    fold = dataset[-1]  # last letter of dataset
    paths_dict = delta_batcher.get_lsms_tfrecord_pairs(incountry_folds[fold])

    num_train = len(paths_dict['train'])
    num_val = len(paths_dict['val'])
    print('Train pairs:', num_train)
    print('Val pairs:', num_val)
    train_steps_per_epoch = int(np.ceil(num_train / batch_size))
    val_steps_per_epoch = int(np.ceil(num_val / batch_size))

    def get_batcher(tfrecord_pairs, shuffle, augment, epochs, cache, orig_labels):
        kwargs = dict(
            tfrecord_pairs=tfrecord_pairs,
            dataset='LSMS',
            batch_size=batch_size,
            label_name=label_name,
            num_threads=num_threads,
            epochs=epochs,
            ls_bands=ls_bands,
            nl_band=nl_band,
            shuffle=shuffle,
            augment=augment,
            normalize=True,
            cache=cache)
        if batcher == 'delta':
            kwargs['orig_labels'] = orig_labels
        return Batcher(**kwargs)

    train_tfrecord_pairs_ph = tf.placeholder(paths_dict['train'].dtype, paths_dict['train'].shape)
    val_tfrecord_pairs_ph = tf.placeholder(paths_dict['val'].dtype, paths_dict['val'].shape)

    with tf.name_scope('train_batcher'):
        train_batcher = get_batcher(
            train_tfrecord_pairs_ph,
            shuffle=True,
            augment=augment,
            epochs=max_epochs,
            cache='train' in cache,
            orig_labels=orig_labels)
        train_init_iter, train_batch = train_batcher.get_batch()

    with tf.name_scope('train_eval_batcher'):
        train_eval_batcher = get_batcher(
            train_tfrecord_pairs_ph,
            shuffle=False,
            augment=False,
            epochs=max_epochs + 1,  # may need extra epoch at the end of training
            cache='train_eval' in cache,
            orig_labels=orig_labels)
        train_eval_init_iter, train_eval_batch = train_eval_batcher.get_batch()

    with tf.name_scope('val_batcher'):
        val_batcher = get_batcher(
            val_tfrecord_pairs_ph,
            shuffle=False,
            augment=False,
            epochs=max_epochs + 1,  # may need extra epoch at the end of training
            cache='val' in cache,
            orig_labels=orig_labels)
        val_init_iter, val_batch = val_batcher.get_batch()

    # ====================
    #        MODEL
    # ====================
    print('Building model...')
    if orig_labels or batcher == 'deltaclass':
        model_params['num_outputs'] = 3
    else:
        model_params['num_outputs'] = 1

    with tf.variable_scope(tf.get_variable_scope()) as model_scope:
        train_model = model_class(train_batch['images'], is_training=True, **model_params)
        train_preds = tf.squeeze(train_model.outputs, name='train_preds')

    with tf.variable_scope(model_scope, reuse=True):
        train_eval_model = model_class(train_eval_batch['images'], is_training=False, **model_params)
        train_eval_preds = tf.squeeze(train_eval_model.outputs, name='train_eval_preds')

    with tf.variable_scope(model_scope, reuse=True):
        val_model = model_class(val_batch['images'], is_training=False, **model_params)
        val_preds = tf.squeeze(val_model.outputs, name='val_preds')

    trainer = Trainer(
        train_batch, train_eval_batch, val_batch,
        train_model, train_eval_model, val_model,
        train_preds, train_eval_preds, val_preds,
        sess, train_steps_per_epoch, ls_bands, nl_band, learning_rate, lr_decay,
        log_dir, save_ckpt_dir, init_ckpt_dir, imagenet_weights_path,
        hs_weight_init, exclude_final_layer, image_summaries=False)

    # initialize the training dataset iterator
    sess.run([train_init_iter, train_eval_init_iter, val_init_iter], feed_dict={
        train_tfrecord_pairs_ph: paths_dict['train'],
        val_tfrecord_pairs_ph: paths_dict['val']
    })

    for epoch in range(max_epochs):
        if epoch % eval_every == 0:
            trainer.eval_train(max_nbatches=train_steps_per_epoch)
            trainer.eval_val(max_nbatches=val_steps_per_epoch)
        trainer.train_epoch(print_every)

    trainer.eval_train(max_nbatches=train_steps_per_epoch)
    trainer.eval_val(max_nbatches=val_steps_per_epoch)

    csv_log_path = os.path.join(log_dir, 'results.csv')
    trainer.log_results(csv_log_path)


def run_training_wrapper(**params):
    '''
    params is a dict with keys matching the FLAGS defined below
    '''
    # print all of the flags
    pprint(params)

    # parameters that might be 'None'
    none_params = ['ls_bands', 'nl_band', 'exclude_final_layer', 'hs_weight_init',
                   'imagenet_weights_path', 'init_ckpt_dir']
    for p in none_params:
        if params[p] == 'None':
            params[p] = None

    # reset any existing graph
    tf.reset_default_graph()

    # set the random seeds
    seed = params['seed']
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # create the log and checkpoint directories if needed
    full_experiment_name = get_full_experiment_name(
        params['experiment_name'], params['batch_size'],
        params['fc_reg'], params['conv_reg'], params['lr'])
    log_dir, ckpt_prefix = make_log_and_ckpt_dirs(
        params['log_dir'], params['ckpt_dir'], full_experiment_name)
    print(f'Checkpoint prefix: {ckpt_prefix}')

    params_filepath = os.path.join(log_dir, 'params.txt')
    assert not os.path.exists(params_filepath), f'Stopping. Found previous run at: {params_filepath}'
    with open(params_filepath, 'w') as f:
        pprint(params, stream=f)
        pprint(f'Checkpoint prefix: {ckpt_prefix}', stream=f)

    # Create session
    # - MUST set up os.environ['CUDA_VISIBLE_DEVICES'] before creating the tf.Session object
    if params['gpu_usage'] == 0:  # restrict to CPU only
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(params['gpu'])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model_params = {
        'fc_reg': params['fc_reg'],
        'conv_reg': params['conv_reg'],
        'use_dilated_conv_in_first_layer': False,
    }

    if params['model_name'] == 'resnet':
        model_params['num_layers'] = params['num_layers']

    run_training(
        sess=sess,
        ooc=params['ooc'],
        batcher=params['batcher'],
        dataset=params['dataset'],
        keep_frac=params['keep_frac'],
        model_name=params['model_name'],
        model_params=model_params,
        batch_size=params['batch_size'],
        ls_bands=params['ls_bands'],
        nl_band=params['nl_band'],
        label_name=params['label_name'],
        orig_labels=params['orig_labels'],
        augment=params['augment'],
        learning_rate=params['lr'],
        lr_decay=params['lr_decay'],
        max_epochs=params['max_epochs'],
        print_every=params['print_every'],
        eval_every=params['eval_every'],
        num_threads=params['num_threads'],
        cache=params['cache'],
        log_dir=log_dir,
        save_ckpt_dir=ckpt_prefix,
        init_ckpt_dir=params['init_ckpt_dir'],
        imagenet_weights_path=params['imagenet_weights_path'],
        hs_weight_init=params['hs_weight_init'],
        exclude_final_layer=params['exclude_final_layer'])
    sess.close()


def main(_):
    params = {
        key: flags.FLAGS.__getattr__(key)
        for key in dir(flags.FLAGS)
    }
    run_training_wrapper(**params)


if __name__ == '__main__':
    flags = tf.app.flags
    root = '/atlas/u/chrisyeh/africa_poverty/'

    # paths
    flags.DEFINE_string('experiment_name', 'new_experiment', 'name of the experiment being run')
    flags.DEFINE_string('ckpt_dir', os.path.join(root, 'ckpts/'), 'checkpoint directory')
    flags.DEFINE_string('log_dir', os.path.join(root, 'logs/'), 'log directory')

    # initialization
    flags.DEFINE_string('init_ckpt_dir', None, 'path to checkpoint prefix from which to initialize weights (default None)')
    flags.DEFINE_string('imagenet_weights_path', None, 'path to ImageNet weights for initialization (default None)')
    flags.DEFINE_string('hs_weight_init', None, 'method for initializing weights of non-RGB bands in 1st conv layer, one of [None (default), "random", "same", "samescaled"]')
    flags.DEFINE_boolean('exclude_final_layer', None, 'whether to use checkpoint to initialize final layer (default None)')

    # learning parameters
    flags.DEFINE_string('label_name', 'wealthpooled', 'name of label to use from the TFRecord files')
    flags.DEFINE_boolean('orig_labels', False, 'whether to include original labels for multi-task training')
    flags.DEFINE_integer('batch_size', 64, 'batch size')
    flags.DEFINE_boolean('augment', True, 'whether to use data augmentation')
    flags.DEFINE_float('fc_reg', 1e-3, 'regularization penalty factor for fully connected layers')
    flags.DEFINE_float('conv_reg', 1e-3, 'regularization penalty factor for convolution layers')
    flags.DEFINE_float('lr', 1e-3, 'learning rate')
    flags.DEFINE_float('lr_decay', 1.0, 'decay rate of the learning rate (default 1.0 for no decay)')

    # high-level model control
    flags.DEFINE_string('model_name', 'resnet', 'name of the model to be used, currently only "resnet" is supported')

    # resnet-only params
    flags.DEFINE_integer('num_layers', 18, 'number of ResNet layers, one of [18 (default), 34, 50]')

    # data params
    flags.DEFINE_string('batcher', 'delta', 'batcher type, one of ["delta", "deltaclass"]')
    flags.DEFINE_string('dataset', 'LSMSDeltaIncountryA', 'dataset to use, options depend on batcher (default "LSMSDeltaIncountryA")')
    flags.DEFINE_boolean('ooc', False, 'whether to use out-of-country split (default False)')
    flags.DEFINE_float('keep_frac', 1.0, 'fraction of training data to use (default 1.0)')
    flags.DEFINE_string('ls_bands', None, 'Landsat bands to use, one of [None (default), "rgb", "ms"]')
    flags.DEFINE_string('nl_band', None, 'nightlights band, one of [None (default), "merge", "split"]')

    # system
    flags.DEFINE_integer('gpu', 0, 'which GPU to use (default 0)')
    flags.DEFINE_float('gpu_usage', 0.96, 'GPU memory usage as a percentage, 0 for CPU-only (default 0.96)')
    flags.DEFINE_integer('num_threads', 1, 'number of threads for batcher (default 1)')
    flags.DEFINE_list('cache', [], 'comma-separated list (no spaces) of datasets to cache in memory, choose from [None, "train", "train_eval", "val"]')

    # Misc
    flags.DEFINE_integer('max_epochs', 200, 'maximum number of epochs for training (default 200)')
    flags.DEFINE_integer('eval_every', 1, 'evaluate model on validation set after every so many epochs of training')
    flags.DEFINE_integer('print_every', 10, 'print training statistics after every so many steps')
    flags.DEFINE_integer('seed', 123, 'seed for random initialization and shuffling')

    tf.app.run()
