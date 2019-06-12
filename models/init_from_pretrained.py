import re

import numpy as np
import tensorflow as tf


def get_saved_var_name(model_var, bottleneck=False):
    '''Gets the saved variable's name (from TensorPack) for the given model variable.

    Args
    - model_var: tf.Variable
    - bottleneck: bool

    Returns: str, the saved variable name if the model variable is part of the ResNet,
        None, otherwise.
    '''
    saved_var_name = model_var.name

    # strip the variable name of everything up to and including '*resnet/'
    s = re.search(r'.*resnet/', saved_var_name)
    if s is None:  # if 'resnet/' not in variable name, then it isn't part of the resnet
        return None
    else:
        saved_var_name = saved_var_name.replace(s.group(0), '')

    # replace model variable names with the saved variable counterparts
    conversion = {
        'batch_normalization': 'bn',
        'weights:0': 'W:0',
        'shortcut': 'convshortcut',
        'scale1': 'conv0'  # 'scale1' is a special case, because it has no blocks or subblocks
    }

    if bottleneck:
        conversion['/a/'] = '/conv1/'
        conversion['/b/'] = '/conv2/'
        conversion['/c/'] = '/conv3/'
    else:
        conversion['/A/'] = '/conv1/'
        conversion['/B/'] = '/conv2/'

    for model_str, saved_str in conversion.items():
        saved_var_name = saved_var_name.replace(model_str, saved_str)

    # shift all of the 'block' numbers down by 1
    # search for 'block###', return the '###' part
    s = re.search(r'block(\d+)', saved_var_name)  # search for 'scale###'
    if s is not None:
        block_str = s.group(0)  # block_str = 'block###'
        block_num = int(s.group(1))  # extract the '###' part from 'scale###' and convert it to int
        new_block_str = 'block' + str(block_num - 1)
        saved_var_name = saved_var_name.replace(block_str, new_block_str)

    # shift all of the 'scale' numbers down by 2, then rename to 'group'
    # - NOTE: we already dealt with scale1 above since it is a special case (no blocks or subblocks)
    #   so we don't need to worry about negative numbers here
    s = re.search(r'scale(\d+)', saved_var_name)  # search for 'scale###'
    if s is not None:
        scale_str = s.group(0)  # scale_str = 'scale###'
        scale_num = int(s.group(1))  # extract the '###' part from 'scale###' and convert it to int
        new_group_str = 'group' + str(scale_num - 2)
        saved_var_name = saved_var_name.replace(scale_str, new_group_str)

    return saved_var_name


def init_first_layer_weights(var, rgb_weights, sess, hs_weight_init):
    '''Initializes the weights for filters in the first conv layer

    'resnet/scale1/weights:0' for ResNet
    'vggf/conv1/conv1_weights:0' for VGGF

    If we are using RGB-only, then just initializes var to rgb_weights. Otherwise, uses
    hs_weight_init to determine how to initialize the weights for non-RGB bands.

    Args
    - var: tf.Variable, the filters in the 1st convolution layer, shape [F, F, C, 64]
        - F is the filter size (7 for ResNet, 11 for VGGF)
        - C is either 3 (RGB), 7 (lxv3), or 9 (Landsat7)
    - rgb_weights: ndarray of np.float32, shape [F, F, 3, 64]
    - sess: tf.Session
    - hs_weight_init: str, one of ['random', 'same', 'samescaled']
    '''
    var_shape = np.asarray(var.get_shape().as_list())
    rgb_weights_shape = np.asarray(rgb_weights.shape)

    # only weights in the 1st conv layer need to be adjusted for dealing with hyperspectral images
    # check that the filter shape and num_filters match up, and that RGB weights have 3 channels
    if 'scale1/weights:0' in var.name:  # ResNet
        F = 7
    elif 'conv1/conv1_weights:0' in var.name:  # VGGF
        F = 11
    else:
        raise ValueError('var is not the weights for the first conv layer')

    assert np.all(var_shape[[0, 1]] == [F, F])
    assert np.all(var_shape[[0, 1, 3]] == rgb_weights_shape[[0, 1, 3]])
    assert rgb_weights.shape[2] == 3
    assert rgb_weights.dtype == np.float32

    # if we are using the RGB-only model, then just initialize to saved weights
    if var_shape[2] == 3:
        print('Using rgb only model')
        sess.run(var.assign(rgb_weights))
        return

    # Set up the initializer function
    print('Initializing var different from saved rgb weights:', var.name, ' With shape:', var_shape)
    print('Using ' + hs_weight_init + ' initialization for hyperspectral weights.')
    num_hs_channels = var_shape[2] - rgb_weights.shape[2]
    hs_weights_shape = [F, F, num_hs_channels, 64]

    if hs_weight_init == 'random':
        # initialize the weights in the hyperspectral bands to gaussian with same overall mean and
        # stddev as the RGB channels
        rgb_mean = np.mean(rgb_weights)
        rgb_std = np.std(rgb_weights)
        hs_weights = tf.truncated_normal(hs_weights_shape, mean=rgb_mean, stddev=rgb_std, dtype=tf.float32)
    elif hs_weight_init == 'same':
        # initialize the weight for each position in each filter to the average of the 3 RGB weights
        # at the same position in the same filter
        rgb_mean = rgb_weights.mean(axis=2, keepdims=True)  # shape [F, F, 1, 64]
        hs_weights = np.tile(rgb_mean, (1, 1, num_hs_channels, 1))
    elif hs_weight_init == 'samescaled':
        # similar to hs_weight_init == 'same', but we normalize the weights
        rgb_mean = rgb_weights.mean(axis=2, keepdims=True)  # shape [F, F, 1, 64]
        hs_weights = np.tile(rgb_mean, (1, 1, num_hs_channels, 1))
        rgb_weights *= 3 / (3 + num_hs_channels)
        hs_weights *= 3 / (3 + num_hs_channels)
    else:
        raise ValueError(f'Unknown hs_weight_init type: {hs_weight_init}')

    final_weight = tf.concat([rgb_weights, hs_weights], axis=2)
    print('Shape of 1st layer weights:', final_weight.shape)  # should be (F, F, C, 64)

    sess.run(var.assign(final_weight))


def init_resnet_v2_from_numpy(path, sess, bottleneck=False, hs_weight_init='random'):
    '''
    Args
    - path: str, path to .npz file containing pre-trained weights
    - sess: tf.Session
    - bottleneck: bool
    - hs_weight_init: str, one of ['random', 'same', 'samescaled']
    '''
    saved_weights = np.load(path)

    for model_var in tf.trainable_variables():
        saved_var_name = get_saved_var_name(model_var, bottleneck=bottleneck)
        if (saved_var_name is None) or (saved_var_name not in saved_weights):
            print('Did not find saved value for variable:', model_var.name)
            print('Will use default initalization instead.')
            continue

        saved_var = saved_weights[saved_var_name]
        if 'scale1/weights:0' in model_var.name:
            init_first_layer_weights(model_var, saved_var, sess, hs_weight_init)
        else:
            sess.run(model_var.assign(saved_var))
