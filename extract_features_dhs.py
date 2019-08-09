from batchers import batcher
# from models.vggf_model import VGGF
# from models.simple_cnn import SimpleCNN
# from models.resnet_combo import ResnetCombo
from models.resnet_model import Hyperspectral_Resnet
from utils.run import check_existing, run_extraction_on_models

from collections import defaultdict
from glob import glob
import os
from pprint import pprint
import re

import numpy as np
import tensorflow as tf


ROOT_DIR = '/atlas/u/chrisyeh/africa_poverty/'


# ====================
#      Parameters
# ====================
DATASET_NAME = '2009-17'
BATCH_SIZE = 128
KEEP_FRAC = 1.0
LABEL_NAME = 'wealthpooled'
IS_TRAINING = False

DEFAULT_MODEL_TYPE = 'resnet'

# NAME = 'DHSIncountry'
NAME = 'DHS_OOC'

CKPTS_ROOT_DIR = os.path.join(ROOT_DIR, f'ckpts/{NAME}/')
LOGS_ROOT_DIR = os.path.join(ROOT_DIR, f'logs/{NAME}/')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_bands(bands: str):
    return {
        'ms': ('ms', None),
        'msnl': ('ms', 'split'),
        'nl': (None, 'split'),
        'rgb': ('rgb', None),
        'rgbnl': ('rgb', 'split'),
    }[bands]


# models to run
ALL_MODELS = {}

# Best ResNet-18 transfer models (FINAL)
TRANSFER_MODELS = {
    'ResNet-18 RGB Transfer': {
        'model_dir': 'transfer_2009-17nl_nlcenter_18preact_rgb_b64_fc001_conv001_lr0001',
        'bands': ('rgb', None)},
    'ResNet-18 MS Transfer': {
        'model_dir': 'transfer_2009-17nl_nlcenter_18preact_ms_b64_fc001_conv001_lr0001',
        'bands': ('ms', None)},
}
# ALL_MODELS.update(TRANSFER_MODELS)

# ImageNet 'Transfer' Learning
IMAGENET_TRANSFER_MODELS = []
[
    '18preact_rgb_random',
    '18preact_rgb_random2',
    '18preact_rgb_random3',
    '18preact_rgb_same',
    '18preact_rgbnl_random',
    '18preact_rgbnl_random2',
    '18preact_rgbnl_random3',
    '18preact_rgbnl_same',
    '18preact_rgbnl_samecaled',
    '18preact_ms_random',
    '18preact_ms_random2',
    '18preact_ms_random3',
    '18preact_ms_same',
    '18preact_ms_samecaled',
    '18preact_msnl_random',
    '18preact_msnl_random2',
    '18preact_msnl_random3',
    '18preact_msnl_same',
    '18preact_msnl_samecaled',
]

for model_dir in IMAGENET_TRANSFER_MODELS:
    regex = r'18preact_(\w+)_(\w+)'
    bands_name, init = re.match(regex, model_dir).groups()
    bands_tup = get_bands(bands_name)
    model_name = f'Resnet-18 {bands_name} Init {init}'
    ALL_MODELS[model_name] = {
        'model_dir': model_dir,
        'bands': bands_tup
    }

# Best ResNet-18 OOC End-to-End models
OOC_MODEL_DIRS = []
[
    # 6/14/2019
    '2009-17A_18preact_ms_samescaled_b64_fc01_conv01_lr0001',
    '2009-17B_18preact_ms_samescaled_b64_fc001_conv001_lr0001',
    '2009-17C_18preact_ms_samescaled_b64_fc001_conv001_lr001',
    '2009-17D_18preact_ms_samescaled_b64_fc001_conv001_lr01',
    '2009-17E_18preact_ms_samescaled_b64_fc01_conv01_lr001',

    # 10/7/2018
    '2009-17A_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    '2009-17B_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    '2009-17C_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    '2009-17D_18preact_nl_random_b64_fc1.0_conv1.0_lr01',
    '2009-17E_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',

    # 10/7/2018
    '2009-17A_18preact_rgb_same_b64_fc001_conv001_lr01',
    '2009-17B_18preact_rgb_same_b64_fc001_conv001_lr0001',
    '2009-17C_18preact_rgb_same_b64_fc001_conv001_lr0001',
    '2009-17D_18preact_rgb_same_b64_fc1.0_conv1.0_lr01',
    '2009-17E_18preact_rgb_same_b64_fc001_conv001_lr0001',
]

for model_dir in OOC_MODEL_DIRS:
    regex = r'2009-17(\w)_18preact_(\w+)_\w+_b64.+'
    fold, bands_name = re.match(regex, model_dir).groups()
    bands_tup = get_bands(bands_name)
    model_name = f'Resnet-18 {bands_name} {fold}'
    ALL_MODELS[model_name] = {
        'model_dir': model_dir,
        'bands': bands_tup
    }

# Incountry Models
INCOUNTRY_MODEL_DIRS = []
[
    # 6/12/2019
    'incountryA_18preact_ms_samescaled_b64_fc01_conv01_lr001',
    'incountryB_18preact_ms_samescaled_b64_fc1_conv1_lr001',
    'incountryC_18preact_ms_samescaled_b64_fc1.0_conv1.0_lr0001',
    'incountryD_18preact_ms_samescaled_b64_fc001_conv001_lr0001',
    'incountryE_18preact_ms_samescaled_b64_fc001_conv001_lr0001',

    # May 2019
    'incountryA_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    'incountryB_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    'incountryC_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    'incountryD_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    'incountryE_18preact_nl_random_b64_fc01_conv01_lr001',
]

for model_dir in INCOUNTRY_MODEL_DIRS:
    regex = r'incountry(\w)_18preact_(\w+)_\w+_b64.+'
    fold, bands_name = re.match(regex, model_dir).groups()
    bands_tup = get_bands(bands_name)
    model_name = f'{NAME} Resnet-18 Incountry {bands_name} {fold}'
    ALL_MODELS[model_name] = {
        'model_dir': model_dir,
        'bands': bands_tup
    }

KEEP_MODEL_DIRS = sorted(glob(os.path.join(LOGS_ROOT_DIR, '2009-17*_18preact_nl_random_keep*seed*')))

for model_dir in KEEP_MODEL_DIRS:
    model_dir = os.path.basename(model_dir)
    regex = r'2009-17(\w)_18preact_(\w+)_\w+_keep(.+)_seed(\w+)_b64.+'
    fold, bands_name, keep, seed = re.match(regex, model_dir).groups()
    bands_tup = get_bands(bands_name)
    model_name = f'Resnet-18 {bands_name} {fold}, keep{keep} seed{seed}'
    ALL_MODELS[model_name] = {
        'model_dir': model_dir,
        'bands': bands_tup
    }

MODEL_PARAMS = {
    'fc_reg': 5e-3,  # this doesn't actually matter
    'conv_reg': 5e-3,  # this doesn't actually matter
    'num_layers': 18,
    'num_outputs': 1,
    'is_training': IS_TRAINING,
}

# ====================
# End Parameters
# ====================


def get_model_class(model_type: str):
    if model_type == 'resnet':
        model_class = Hyperspectral_Resnet
    elif model_type == 'vggf':
        model_class = VGGF
    elif model_type == 'simplecnn':
        model_class = SimpleCNN
    elif model_type == 'resnetcombo':
        model_class = ResnetCombo
    else:
        raise ValueError('Unknown model_name. Was not one of ["resnet", "vggf", "simplecnn", "resnetcombo"].')
    return model_class


def get_batcher(ls_bands: str, nl_band: str, num_epochs: int):
    '''
    Args
    - ls_bands: one of [None, 'ms', 'rgb']
    - nl_band: one of [None, 'merge', 'split']
    - num_epochs: int

    Returns
    - b: Batcher
    - size: int, length of dataset
    - feed_dict: dict, feed_dict for initializing the dataset iterator
    '''
    tfrecord_paths = np.asarray(batcher.get_tfrecord_paths(DATASET_NAME, 'all'))
    size = len(tfrecord_paths)
    tfrecord_paths_ph = tf.placeholder(tf.string, shape=[size])
    feed_dict = {tfrecord_paths_ph: tfrecord_paths}

    b = batcher.Batcher(
        tfrecord_files=tfrecord_paths,
        dataset=DATASET_NAME,
        batch_size=BATCH_SIZE,
        label_name=LABEL_NAME,
        num_threads=4,
        epochs=num_epochs,
        ls_bands=ls_bands,
        nl_band=nl_band,
        shuffle=False,
        augment=False,
        normalize=True,
        cache=(num_epochs > 1))
    return b, size, feed_dict


def main():
    # If any *.npz files already exist, print them out then throw an error.
    print('Checking all models for valid checkpoints and no existing *.npz files ...')
    pprint(list(ALL_MODELS.keys()))
    if not check_existing(ALL_MODELS,
                          logs_root_dir=LOGS_ROOT_DIR,
                          ckpts_root_dir=CKPTS_ROOT_DIR,
                          save_filename='feaures.npz'):
        print('Stopping')
        return
    print('Ready to go.')

    # {
    #    config1: [model_info1, model_info2, ...],
    #    ...
    # }
    # where config = (ls_bands, nl_band, model_type)
    models_by_config = defaultdict(list)
    for model_name in ALL_MODELS:
        model_info = ALL_MODELS[model_name]
        ls_bands, nl_band = model_info['bands']
        model_type = model_info.get('model_type', DEFAULT_MODEL_TYPE)
        config = (ls_bands, nl_band, model_type)
        models_by_config[config].append(model_info)

    for config, model_infos in models_by_config.items():
        ls_bands, nl_band, model_type = config
        print('====== Current Config: ======')
        print('- ls_bands:', ls_bands)
        print('- nl_band:', nl_band)
        print('- model_type:', model_type)
        print('- number of models:', len(model_infos))
        print()

        b, size, feed_dict = get_batcher(ls_bands=ls_bands, nl_band=nl_band,
                                         num_epochs=len(model_infos))
        batches_per_epoch = int(np.ceil(size / BATCH_SIZE))

        run_extraction_on_models(
            model_infos,
            ModelClass=get_model_class(model_type),
            model_params=MODEL_PARAMS,
            batcher=b,
            batches_per_epoch=batches_per_epoch,
            logs_root_dir=LOGS_ROOT_DIR,
            ckpts_root_dir=CKPTS_ROOT_DIR,
            save_filename='features.npz',
            batch_keys=['labels', 'locs', 'years'],
            feed_dict=feed_dict)


if __name__ == '__main__':
    main()
