"""
Copyright 2020-2021 Purvang Lapsiwala
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from functools import reduce

import custom_layers as layers
import keras
import tensorflow as tf

MOMENTUM = 0.997
EPSILON = 1e-4


def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = keras.layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                      use_bias=True, name=f'{name}')
    f2 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/bn')
    # f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


class wBiFPNAdd(keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(wBiFPNAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(wBiFPNAdd, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config


def build_wBiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = C3
        P4_in = C4
        P5_in = C5
        P6_in = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
        # P6_in = BatchNormalization(freeze=freeze_bn, name='resample_p6/bn')(P6_in)
        P6_in = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
        P7_in = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
        P7_U = keras.layers.UpSampling2D()(P7_in)
        P6_in = layers.UpsampleLike()([P6_in, P7_U])

        P6_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        P5_in_1 = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                      name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                                  name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        # P5_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        P6_U = keras.layers.UpSampling2D()(P6_td)
        P5_in_1 = layers.UpsampleLike()([P5_in_1, P6_U])

        P5_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in_1, P6_U])
        P5_td = keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P4_in_1 = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                      name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
        P4_in_1 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                                  name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        # P4_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        P5_U = keras.layers.UpSampling2D()(P5_td)
        P4_in_1 = layers.UpsampleLike()([P4_in_1, P5_U])

        P4_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in_1, P5_U])
        P4_td = keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P3_in = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                    name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
        P3_in = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                                name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        # P3_in = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        P4_U = keras.layers.UpSampling2D()(P4_td)
        P3_in = layers.UpsampleLike()([P3_in, P4_U])

        P3_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P4_in_2 = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                      name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                                  name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        # P4_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        P3_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_in_2 = layers.UpsampleLike()([P4_in_2, P3_D])
        P4_td = layers.UpsampleLike()([P4_td, P3_D])

        P4_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in_2, P4_td, P3_D])
        P4_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P5_in_2 = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                      name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                                  name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        # P5_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        P4_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_in_2 = layers.UpsampleLike()([P5_in_2, P4_D])
        P5_td = layers.UpsampleLike()([P5_td, P4_D])

        P5_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in_2, P5_td, P4_D])
        P5_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        P5_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_in = layers.UpsampleLike()([P6_in, P5_D])
        P6_td = layers.UpsampleLike()([P6_td, P5_D])

        P6_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_in = layers.UpsampleLike()([P7_in, P6_D])

        P7_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        P7_U = keras.layers.UpSampling2D()(P7_in)
        P6_in = layers.UpsampleLike()([P6_in, P7_U])

        P6_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        P6_U = keras.layers.UpSampling2D()(P6_td)
        P5_in = layers.UpsampleLike()([P5_in, P6_U])

        P5_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U])
        P5_td = keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P5_U = keras.layers.UpSampling2D()(P5_td)
        P4_in = layers.UpsampleLike()([P4_in, P5_U])

        P4_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U])
        P4_td = keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P4_U = keras.layers.UpSampling2D()(P4_td)
        P3_in = layers.UpsampleLike()([P3_in, P4_U])

        P3_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P3_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_in = layers.UpsampleLike()([P4_in, P3_D])
        P4_td = layers.UpsampleLike()([P4_td, P3_D])

        P4_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
        P4_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P4_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_in = layers.UpsampleLike()([P5_in, P4_D])
        P5_td = layers.UpsampleLike()([P5_td, P4_D])

        P5_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
        P5_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        P5_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_in = layers.UpsampleLike()([P6_in, P5_D])
        P6_td = layers.UpsampleLike()([P6_td, P5_D])

        P6_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_in = layers.UpsampleLike()([P7_in, P6_D])

        P7_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)
    return P3_out, P4_td, P5_td, P6_td, P7_out


def build_BiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = C3
        P4_in = C4
        P5_in = C5
        P6_in = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
        # P6_in = BatchNormalization(freeze=freeze_bn, name='resample_p6/bn')(P6_in)
        P6_in = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
        P7_in = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
        P7_U = keras.layers.UpSampling2D()(P7_in)
        P6_in = layers.UpsampleLike()([P6_in, P7_U])
        P6_td = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        P5_in_1 = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                      name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                                  name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        # P5_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        P6_U = keras.layers.UpSampling2D()(P6_td)
        P5_in_1 = layers.UpsampleLike()([P5_in_1, P6_U])

        P5_td = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in_1, P6_U])
        P5_td = keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P4_in_1 = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                      name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
        P4_in_1 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                                  name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        # P4_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        P5_U = keras.layers.UpSampling2D()(P5_td)
        P4_in_1 = layers.UpsampleLike()([P4_in_1, P5_U])

        P4_td = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in_1, P5_U])
        P4_td = keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P3_in = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                    name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
        P3_in = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                                name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        # P3_in = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        P4_U = keras.layers.UpSampling2D()(P4_td)

        P3_in = layers.UpsampleLike()([P3_in, P4_U])

        P3_out = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
        # P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
        #                             name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)

        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='P3')(P3_out)

        P4_in_2 = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                      name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                                  name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        # P4_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        P3_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_in_2 = layers.UpsampleLike()([P4_in_2, P3_D])
        P4_td = layers.UpsampleLike()([P4_td, P3_D])

        P4_out = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in_2, P4_td, P3_D])
        P4_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
        # P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
        #                             name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='P4')(P4_out)

        P5_in_2 = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                      name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                                  name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        # P5_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        P4_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)

        P5_in_2 = layers.UpsampleLike()([P5_in_2, P4_D])
        P5_td = layers.UpsampleLike()([P5_td, P4_D])

        P5_out = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in_2, P5_td, P4_D])
        P5_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
        # P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
        #                             name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='P5')(P5_out)

        P5_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_in = layers.UpsampleLike()([P6_in, P5_D])
        P6_td = layers.UpsampleLike()([P6_td, P5_D])
        P6_out = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
        # P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
        #                             name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='P6')(P6_out)

        P6_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_in = layers.UpsampleLike()([P7_in, P6_D])
        P7_out = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
        # P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
        #                             name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name='P7')(P7_out)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        P7_U = keras.layers.UpSampling2D()(P7_in)
        P7_U = layers.UpsampleLike()([P7_U, P6_in])

        P6_td = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        P6_U = keras.layers.UpSampling2D()(P6_td)
        P6_U = layers.UpsampleLike()([P6_U, P5_in])

        P5_td = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U])
        P5_td = keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P5_U = keras.layers.UpSampling2D()(P5_td)
        P5_U = layers.UpsampleLike()([P5_U, P4_in])

        P4_td = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U])
        P4_td = keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P4_U = keras.layers.UpSampling2D()(P4_td)
        P4_U = layers.UpsampleLike()([P4_U, P3_in])

        P3_out = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])

        P3_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P3_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)

        P4_td = layers.UpsampleLike()([P4_td, P4_in])
        P3_D = layers.UpsampleLike()([P3_D, P4_in])

        P4_out = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])

        P4_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P4_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_td = layers.UpsampleLike()([P5_td, P5_in])
        P4_D = layers.UpsampleLike()([P4_D, P5_in])

        P5_out = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])

        P5_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        P5_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_td = layers.UpsampleLike()([P6_td, P6_in])
        P5_D = layers.UpsampleLike()([P5_D, P6_in])

        P6_out = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])

        P6_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P6_D = layers.UpsampleLike()([P6_D, P7_in])

        P7_out = keras.layers.Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])

        P7_out = keras.layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

    return P3_out, P4_td, P5_td, P6_td, P7_out


def build_FPN(C3, C4, C5, feature_size=256):
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]


def build_NoFPN(C3, C4, C5, feature_size=256):
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5_conv')(C5)
    P5 = keras.layers.Activation('relu', name='P5')(P5)
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P4_conv')(C4)
    P4 = keras.layers.Activation('relu', name='P4')(P4)

    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P3_conv')(C3)
    P3 = keras.layers.Activation('relu', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6_conv')(C5)
    P6 = keras.layers.Activation('relu', name='P6')(P6)

    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7_conv')(P6)
    P7 = keras.layers.Activation('relu', name='P7')(P7)

    return [P3, P4, P5, P6, P7]



