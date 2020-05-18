##########################################################################
# NNAD (Neural Networks for Automated Driving) training scripts          #
# Copyright (C) 2020 FZI Research Center for Information Technology      #
#                                                                        #
# This program is free software: you can redistribute it and/or modify   #
# it under the terms of the GNU General Public License as published by   #
# the Free Software Foundation, either version 3 of the License, or      #
# (at your option) any later version.                                    #
#                                                                        #
# This program is distributed in the hope that it will be useful,        #
# but WITHOUT ANY WARRANTY; without even the implied warranty of         #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          #
# GNU General Public License for more details.                           #
#                                                                        #
# You should have received a copy of the GNU General Public License      #
# along with this program.  If not, see <https://www.gnu.org/licenses/>. #
##########################################################################

import tensorflow as tf
import tensorflow_addons as tfa
from .constants import *
from .Resnet import *

class FlowUpsample(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)

        self.transposed_conv = tf.keras.layers.Conv2DTranspose(2,
            (4, 4),
            (2, 2),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='transposed_conv')

    def call(self, x):
        x = self.transposed_conv(x)
        # If we upsample the flow image, we also have to scale the flow vectors:
        x *= tf.constant(2.0)
        return x

class FeatureDownsample(tf.keras.Model):
    def __init__(self, name, num_output_channels):
        super().__init__(name=name)

        self.bn = Normalization()

        self.conv = tf.keras.layers.SeparableConv2D(num_output_channels,
            (3, 3),
            padding='same',
            strides=(2, 2),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
            name='downsample')

    def call(self, x, train_batch_norm=False):
        x = self.conv(x)
        x = self.bn(x, training=train_batch_norm)
        x = tf.keras.activations.relu(x, 0.1)
        return x

class FlowEstimator(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)

        self.rm1 = ResnetModule('rm1', [64, 64, 128])
        self.rm2 = ResnetModule('rm2', [32, 32, 64])
        self.conv = tf.keras.layers.SeparableConv2D(2,
            (3, 3),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
            name='conv')

    def call(self, x, train_batch_norm=False):
        x = self.rm1(x, train_batch_norm=train_batch_norm)
        x = self.rm2(x, train_batch_norm=train_batch_norm)
        x = self.conv(x)
        return x

def _warp(features, flow):
    # We need to use the negative flow here because of how tfa.image.dense_image_warp works
    # What we really want is to look up the coordinates at coord_old + flow and map them back to coorld_old
    neg_flow = -flow
    warped = tfa.image.dense_image_warp(features, neg_flow)
    warped.set_shape(features.get_shape())
    return warped

'''
This class estimates the flow between the current and previous images.
'''
class Flow(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)

        self.channel_adjust = tf.keras.layers.Conv2D(32,
            (1, 1),
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l1(L2_REGULARIZER_WEIGHT),
            name='conv_adjust')

        self.fe0 = FlowEstimator('fe0')
        self.fe1 = FlowEstimator('fe1')
        self.fe2 = FlowEstimator('fe2')
        self.fe3 = FlowEstimator('fe3')
        self.fe4 = FlowEstimator('fe4')

        self.feature_downsample0 = FeatureDownsample('feature_downsample0', 64)
        self.feature_downsample1 = FeatureDownsample('feature_downsample1', 96)
        self.feature_downsample2 = FeatureDownsample('feature_downsample2', 128)
        self.feature_downsample3 = FeatureDownsample('feature_downsample3', 196)

        self.flow_upsample1 = FlowUpsample('flow_upsample1')
        self.flow_upsample2 = FlowUpsample('flow_upsample2')
        self.flow_upsample3 = FlowUpsample('flow_upsample3')
        self.flow_upsample4 = FlowUpsample('flow_upsample4')

        self.correlate = tfa.layers.optical_flow.CorrelationCost(1, 4, 1, 1, 4, "channels_last")

    # Calculates _forward_ flow. For backward flow exchange inputs.
    def call(self, inputs, train_batch_norm=False):
        current, prev = inputs
        current = self.channel_adjust(current)
        prev = self.channel_adjust(prev)

        # Downsample
        current_0 = current
        current_1 = self.feature_downsample0(current_0, train_batch_norm=train_batch_norm)
        current_2 = self.feature_downsample1(current_1, train_batch_norm=train_batch_norm)
        current_3 = self.feature_downsample2(current_2, train_batch_norm=train_batch_norm)
        current_4 = self.feature_downsample3(current_3, train_batch_norm=train_batch_norm)
        prev_0 = prev
        prev_1 = self.feature_downsample0(prev_0, train_batch_norm=train_batch_norm)
        prev_2 = self.feature_downsample1(prev_1, train_batch_norm=train_batch_norm)
        prev_3 = self.feature_downsample2(prev_2, train_batch_norm=train_batch_norm)
        prev_4 = self.feature_downsample3(prev_3, train_batch_norm=train_batch_norm)

        # Correlation on level 4
        correlation_4 = self.correlate([prev_4, current_4])
        x = tf.concat([correlation_4, current_4, prev_4], axis=-1)
        flow_4 = self.fe4(x, train_batch_norm=train_batch_norm)
        flow_4_up = self.flow_upsample4(flow_4)

        # Correlation and flow on level 3
        warped_current_3 = _warp(current_3, flow_4_up)
        correlation_3 = self.correlate([prev_3, warped_current_3])
        x = tf.concat([correlation_3, warped_current_3, prev_3, flow_4_up], axis=-1)
        flow_3 = self.fe3(x, train_batch_norm=train_batch_norm)
        flow_3_up = self.flow_upsample3(flow_3)

        # Correlation and flow on level 2
        warped_current_2 = _warp(current_2, flow_3_up)
        correlation_2 = self.correlate([prev_2, warped_current_2])
        x = tf.concat([correlation_2, warped_current_2, prev_2, flow_3_up], axis=-1)
        flow_2 = self.fe2(x, train_batch_norm=train_batch_norm)
        flow_2_up = self.flow_upsample2(flow_2)

        # Correlation and flow on level 1
        warped_current_1 = _warp(current_1, flow_2_up)
        correlation_1 = self.correlate([prev_1, warped_current_1])
        x = tf.concat([correlation_1, warped_current_1, prev_1, flow_2_up], axis=-1)
        flow_1 = self.fe1(x, train_batch_norm=train_batch_norm)
        flow_1_up = self.flow_upsample1(flow_1)

        # Correlation and flow on level 0
        warped_current_0 = _warp(current_0, flow_1_up)
        correlation_0 = self.correlate([prev_0, warped_current_0])
        x = tf.concat([correlation_0, warped_current_0, prev_0, flow_1_up], axis=-1)
        flow_0 = self.fe0(x, train_batch_norm=train_batch_norm)

        results = {}
        results['flow_0'] = flow_0
        results['flow_1_up'] = flow_1_up
        results['flow_2_up'] = flow_2_up
        results['flow_3_up'] = flow_3_up
        results['flow_4_up'] = flow_4_up

        return results

class FlowWarp(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)
        self.channel_reduce = tf.keras.layers.SeparableConv2D(1024,
            (1, 1),
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l1(L2_REGULARIZER_WEIGHT),
            name='conv_reduce')

    def call(self, inputs, train_batch_norm=False):
        x, x_prev, bw_flow = inputs

        x_prev_warped = _warp(x_prev, bw_flow)
        x = tf.concat([x, x_prev_warped, bw_flow], axis=-1)
        x = self.channel_reduce(x)

        return x