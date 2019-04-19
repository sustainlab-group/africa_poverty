import tensorflow as tf

from .resnet_config import Config

# Adapted from https://github.com/ry/tensorflow-resnet/blob/master/resnet.py

BN_DECAY = 0.99
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]
DEFAULT_DTYPE = tf.float32

activation = tf.nn.relu


def update_feature_dict(x, c):
    feature_dict = c['feature_dict']
    if feature_dict is None:
        return
    for k in sorted(feature_dict.keys()):
        if not isinstance(feature_dict[k], bool):
            continue
        if feature_dict[k]:
            feature_dict[k] = tf.reduce_mean(x, axis=[1, 2], name=f'feature_dict_avg_pool_{k}')
            return
        else:
            feature_dict.pop(k)
            return


def inference(x, is_training,
              num_classes=1000,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              use_bias=False,  # defaults to using batch norm
              bottleneck=True,
              use_dilated_conv_in_first_layer=False,
              blocks_to_save=None,
              conv_reg=0.001,
              fc_reg=0.001):
    '''Implements Resnet v2 (preactivation).

    Args
    - x: tf.Tensor, shape [batch_size, H, W, C], type float32
    - is_training: bool
    - num_classes: int, number of output classes for final fully-connected layer,
        set to None if no fully-connected layer is desired
    - num_blocks: list of 4 integers, number of blocks in each of the 4 "groups" (or "scales")
    - use_bias: bool, if True performs conv(x)+bias, if False performs batch_norm(conv(x))
    - bottleneck: bool, if True uses bottleneck layer
    - use_dilated_conv_in_first_layer: bool
    - blocks_to_save: dict of {int: None}, keys are block numbers from which to save features
        NOTE: the keys are BLOCK numbers, not LAYER numbers
    - conv_reg: float, L2 weight regularization penalty for conv layers
    - fc_reg: float, L2 weight regularization penalty for fully-connected layer

    Returns:
    - x: if num_classes is None, x is equal to features_layer
        otherwise, x is a tf.Tensor with shape [batch_size, num_classes]
    - features_layer: tf.Tensor with shape [batch_size, num_final_filters] where num_final_filters
        is the number of filters in the last layer of the resnet before the average-pooling
    '''
    with tf.variable_scope('resnet'):
        c = Config()
        c['bottleneck'] = bottleneck
        c['is_training'] = is_training
        c['ksize'] = 3
        c['stride'] = 1
        c['use_bias'] = use_bias
        c['fc_units_out'] = num_classes
        c['num_blocks'] = num_blocks
        c['conv_reg'] = conv_reg
        c['fc_reg'] = fc_reg
        c['stack_stride'] = 2  # default stride for the 1st conv of the 1st block in each stack
        c['is_first_stack'] = False

        # Make blocks_to_save into a dict of {block_number: bool}
        # that indicates whether or not each block's features are to be saved
        if blocks_to_save is not None:
            valid_keys = range(1, sum(num_blocks) + 1)
            for k in blocks_to_save.keys():
                if k not in valid_keys:
                    raise Exception('Entered invalid block for feature extraction.')
            for i in valid_keys:
                blocks_to_save[i] = (i in blocks_to_save)
        c['feature_dict'] = blocks_to_save

        with tf.variable_scope('scale1'):
            c['conv_filters_out'] = 64
            c['ksize'] = 7
            c['stride'] = 2
            if use_dilated_conv_in_first_layer:
                # Note: Fixed stride of 1 means double w/h for the rest of the network
                x = first_layer_dilated_conv(x, c)
            else:
                x = conv(x, c)
            x = bn_activation(x, c)
            x = tf.identity(x, name='scale1_img')

        with tf.variable_scope('scale2'):
            x = _max_pool(x, ksize=3, stride=2)
            c['num_blocks'] = num_blocks[0]
            c['is_first_stack'] = True
            c['stack_stride'] = 1  # max_pool already reduced input dims, so 1st conv layer here will use stride=1
            c['block_filters_internal'] = 64
            x = stack(x, c)
            x = tf.identity(x, name='scale2_img')

        with tf.variable_scope('scale3'):
            c['num_blocks'] = num_blocks[1]
            c['block_filters_internal'] = 128
            assert c['is_first_stack'] is False
            assert c['stack_stride'] == 2
            x = stack(x, c)
            x = tf.identity(x, name='scale3_img')

        with tf.variable_scope('scale4'):
            c['num_blocks'] = num_blocks[2]
            c['block_filters_internal'] = 256
            x = stack(x, c)
            x = tf.identity(x, name='scale4_img')

        with tf.variable_scope('scale5'):
            c['num_blocks'] = num_blocks[3]
            c['block_filters_internal'] = 512
            x = stack(x, c)
            x = tf.identity(x, name='scale5_img')

        # post-net
        x = tf.reduce_mean(x, axis=[1, 2], name='avg_pool')  # avg pool across image width and height
        features_layer = x

        if num_classes is not None:
            with tf.variable_scope('fc'):
                x = fc(x, c)

        return x, features_layer


def stack(x, c):
    block_fn = block_preact

    # first block in the stack usually performs the downsampling via stride-2 convolution
    with tf.variable_scope('block1'):
        c['block_stride'] = c['stack_stride']
        c['is_first_block_of_first_stack'] = c['is_first_stack']
        x = block_fn(x, c)

    for n in range(2, c['num_blocks'] + 1):
        with tf.variable_scope(f'block{n}'):
            c['block_stride'] = 1
            c['is_first_block_of_first_stack'] = False
            x = block_fn(x, c)
    return x


def block_preact(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed.
    # That is the case when bottleneck=False but when bottleneck=True,
    # filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']
    c['conv_filters_out'] = c['block_filters_internal']

    is_changing_dims = (filters_out != filters_in) or (c['block_stride'] != 1)

    # apply preactivation as needed
    if is_changing_dims:
        # TensorPack claims that input into 1st stack is already "activated"
        if not c['is_first_block_of_first_stack']:
            # common BN, ReLU
            with tf.variable_scope('preact'):
                x = bn_activation(x, c)

        # shortcut needs conv to match dimensions
        with tf.variable_scope('shortcut'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(x, c)
    else:
        shortcut = x

        if not c['is_first_block_of_first_stack']:
            # apply BN + ReLU to non-shortcut branch only
            with tf.variable_scope('preact'):
                x = bn_activation(x, c)

    if c['bottleneck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            x = conv(x, c)
            x = bn_activation(x, c)

        with tf.variable_scope('b'):
            # TensorPack performs the stride-2 downsampling in the 3x3 convolution, even though
            # Kaiming's own implementation suggests that the downsampling should go in the 1x1
            # convolution of scope('a'). We match TensorPack's implementation here so that we
            # can use their pre-trained weights.
            c['stride'] = c['block_stride']
            x = conv(x, c)
            x = bn_activation(x, c)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
            x = bn_activation(x, c)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)

    x = x + shortcut
    update_feature_dict(x, c)
    return x


def bn(x, c):
    if c['use_bias']:
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer())
        return x + bias
    else:
        return tf.layers.batch_normalization(x, momentum=BN_DECAY, training=c['is_training'])


def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=c['fc_reg'])
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype=DEFAULT_DTYPE,
                  trainable=True):
    '''Wrapper around tf.get_variable to do weight decay and add to resnet collection'''
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.variance_scaling_initializer(scale=2.0, mode='fan_out', distribution='normal')
    weights = _get_variable('weights',
                            shape=shape,
                            initializer=initializer,
                            weight_decay=c['conv_reg'])
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def first_layer_dilated_conv(x, c):
    ksize = c['ksize']
    filters_out = c['conv_filters_out']
    filters_in = x.get_shape()[-1]
    if filters_in != 9:
        raise Exception('Attempting to use dilated convolution on image that does not have 9 bands. Is rgb_only True?')

    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            initializer=initializer,
                            weight_decay=c['conv_reg'])
    # sum several convolutions across layers, dilate them so that each conv is looking at the image at its proper resolution.
    # i.e. if each pixel is 15 meters, and an image has a resolution of 30 meters, its convolution
    #     should be dilated with d=2 so that it looks at its image correctly
    # assumes the bands are in the following order
    """
    In order and by index, the bands are:
    0: Blue (Band 1)
    1: Green (Band 2)
    2: Red (Band 3)
    3: Near Infared (NIR) (Band 4)
    4: Short-wave Infrared 1 (SWIR1) (Band 5)
    5: Short-wave Infrared 2 (SWIR2) (Band 7)
    6: Panchromatic (Band 8)
    7: Thermal 1 (Band 6 VCID 1)
    8: Thermal 2 (Band 6 VCID 2)
    """
    _15_meter = [0,1,2,6]
    _30_meter = [3,4,5]
    _60_meter = [7,8]
    split_weights = tf.split(axis=2, num_or_size_splits=9, value=weights)
    split_x = tf.split(axis=3, num_or_size_splits=9, value=x)

    def do_dilated_cov(indicies, rate, name):
        return tf.nn.atrous_conv2d(
            value=tf.concat(axis=3, values=[band for i, band in enumerate(split_x) if i in indicies]),
            filters=tf.concat(axis=2, values=[_filter for i, _filter in enumerate(split_weights) if i in indicies]),
            rate=rate,
            padding='SAME',
            name=name)

    dilated_conv_15m = do_dilated_cov(_15_meter, 1, '15_meter_dilated_conv')
    dilated_conv_30m = do_dilated_cov(_30_meter, 2, '30_meter_dilated_conv')
    dilated_conv_60m = do_dilated_cov(_60_meter, 4, '60_meter_dilated_conv')
    return dilated_conv_15m + dilated_conv_30m + dilated_conv_60m


def _max_pool(x, ksize=3, stride=2):
    x = tf.nn.max_pool(x,
                       ksize=[1, ksize, ksize, 1],
                       strides=[1, stride, stride, 1],
                       padding='SAME')
    return x


def bn_activation(x, c):
    x = bn(x, c)
    x = activation(x)
    return x
