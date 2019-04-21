from models.base_model import BaseModel
from models.init_from_pretrained import init_resnet_v2_from_numpy
from models import hyperspectral_resnet

import tensorflow as tf


class Hyperspectral_Resnet(BaseModel):
    def __init__(self, inputs, num_outputs, is_training,
                 fc_reg=0.0003, conv_reg=0.0003,
                 use_dilated_conv_in_first_layer=False,
                 num_layers=50, blocks_to_save=None):
        '''
        Args
        - inputs: tf.Tensor, shape [batch_size, H, W, C], type float32
        - num_outputs: int, number of output classes
            set to None if we are only extracting features
        - is_training: bool, or tf.placeholder of type tf.bool
        - fc_reg: float, regularization for weights in fully-connected layer
        - conv_reg: float, regularization for weights in conv layers
        - use_dilated_conv_in_first_layer: bool
        - num_layers: int, one of [18, 34, 50]
        - blocks_to_save: list of int, the blocks in the resnet from which to save features
            set to None to not save the outputs of earlier blocks in the resnet
            NOTE: this is a list of BLOCK numbers, not LAYER numbers
        '''
        super(Hyperspectral_Resnet, self).__init__(
            inputs=inputs,
            num_outputs=num_outputs,
            is_training=is_training,
            fc_reg=fc_reg,
            conv_reg=conv_reg)

        # determine bottleneck or not
        if num_layers in [18, 34]:
            self.bottleneck = False
        elif num_layers in [50]:
            self.bottleneck = True
        else:
            raise ValueError('Invalid num_layers passed to model')

        # set num_blocks
        if num_layers == 18:
            num_blocks = [2, 2, 2, 2]
        elif num_layers in [34, 50]:
            num_blocks = [3, 4, 6, 3]
        else:
            raise ValueError('Invalid num_layers passed to model')

        self.block_features = None
        if blocks_to_save is not None:
            self.block_features = {block_index: None for block_index in blocks_to_save}

        # outputs: tf.Tensor, shape [batch_size, num_outputs], type float32
        # features_layer: tf.Tensor, shape [batch_size, num_features], type float32
        self.outputs, self.features_layer = hyperspectral_resnet.inference(
            inputs,
            is_training=is_training,
            num_classes=num_outputs,
            num_blocks=num_blocks,
            bottleneck=self.bottleneck,
            use_dilated_conv_in_first_layer=use_dilated_conv_in_first_layer,
            blocks_to_save=self.block_features,
            conv_reg=conv_reg,
            fc_reg=fc_reg)

    def init_from_numpy(self, path, sess, hs_weight_init='random'):
        '''
        Args
        - path: str, path to a .npz file of pre-trained weights (from TensorPack)
        - sess: tf.Session
        - hs_weight_init: str, one of ['random', 'same'], determines how the weights for non-rgb
            bands in the 1st conv layer are initialized
            - 'random' initializes them using a tf.truncated_normal() initializer with same mean and
                stddev as the RGB bands
            - 'same' initializes them using the mean of the RGB weights
        '''
        init_resnet_v2_from_numpy(path, sess, bottleneck=self.bottleneck,
                                  hs_weight_init=hs_weight_init)

    def get_first_layer_weights(self):
        '''Gets the weights in the first layer of the CNN

        Returns: tf.Tensor, shape [F_height, F_width, F_channels, num_filters]
        '''
        with tf.variable_scope('resnet/scale1', reuse=True):
            return tf.get_variable('weights')

    def get_final_layer_weights(self):
        '''Gets the weights in the final fully-connected layer after the conv layers.

        Returns: list of tf.Tensor
        '''
        return tf.trainable_variables(scope='resnet/fc')

    def get_first_layer_summaries(self, ls_bands=None, nl_band=None):
        '''Creates the following summaries:
        - histogram of weights in 1st conv layer
        - (if model includes batch-norm layer) histogram of 1st batch-norm layer's moving mean

        Args
        - ls_bands: one of [None, 'rgb', 'ms'], if 'ms' then add separate histograms for RGB vs. other
        channel weights the first layer of the CNN
        - nl_band: one of [None, 'split', 'merge']

        Returns
        - summaries: tf.summary, merged summaries
        '''
        summaries = []
        with tf.variable_scope('resnet/scale1', reuse=True):
            x = tf.get_variable('batch_normalization/moving_mean')
        mm_sum = tf.summary.histogram('scale1_moving_mean', x)
        summaries.append(mm_sum)

        x = self.get_first_layer_weights()
        print('First layer weights:', x)
        weights_hist = tf.summary.histogram('scale1_weights_histogram', x)
        summaries.append(weights_hist)

        if ls_bands in ['rgb', 'ms']:
            weights_rgb_hist = tf.summary.histogram('scale1_weights_histogram_RGB', x[:, :, 0:3, :])
            summaries.append(weights_rgb_hist)

        if ls_bands == 'ms':
            weights_ms_hist = tf.summary.histogram('scale1_weights_histogram_MS', x[:, :, 3:7, :])
            summaries.append(weights_ms_hist)

        if nl_band == 'merge':
            weights_nl_hist = tf.summary.histogram('scale1_weights_histogram_NL', x[:, :, -1:, :])
            summaries.append(weights_nl_hist)
        elif nl_band == 'split':
            weights_nl_hist = tf.summary.histogram('scale1_weights_histogram_NL', x[:, :, -2:, :])
            summaries.append(weights_nl_hist)
        else:
            assert nl_band is None

        return tf.summary.merge(summaries)
