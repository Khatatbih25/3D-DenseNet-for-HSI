#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Machine Learning Model module

This script defines machine learning model creation functions.

Author:  Christopher Good
Version: 1.0.0

Usage: models.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###
import math

### Other Library Imports ###
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.layers import (
    Activation,
    AveragePooling3D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv3D,
    Convolution3D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling3D,
    Input,
    MaxPooling3D
)
from tensorflow.keras.models import (
    Model, 
    Sequential,
) 
from tensorflow.keras.optimizers import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    Ftrl,
    Nadam,
    RMSprop,
    SGD,
)

def get_optimizer(**hyperparams):
    """
    Returns appropriately constructed optimizer from hyperparameter
    inputs.

    Parameters
    ----------
    **hyperparams : dict
        dictionary of hyperparameter values to use to construct the
        optimizer

    Returns
    -------
    optimizer : tensorflow.keras.optimizer.Optimizer
        A keras Optimizer object constructed to hyperparam specification
    """
    
    # Get requisite hyperparameter values
    optimizer_name = hyperparams['optimizer']
    learning_rate = hyperparams['lr']
    momentum = hyperparams['momentum']
    epsilon = hyperparams['epsilon']
    initial_accumulator_value = hyperparams['initial_accumulator_value']
    beta_1 = hyperparams['beta_1']
    beta_2 = hyperparams['beta_2']
    amsgrad = hyperparams['amsgrad']
    rho = hyperparams['rho']
    centered = hyperparams['centered']
    nesterov = hyperparams['nesterov']
    learning_rate_power = hyperparams['learning_rate_power']
    l1_regularization_strength = hyperparams['l1_regularization_strength']
    l2_regularization_strength = hyperparams['l2_regularization_strength']
    l2_shrinkage_regularization_strength = hyperparams['l2_shrinkage_regularization_strength']
    beta = hyperparams['beta']

    # Set up the optimizers according to the input hyperparameters
    if optimizer_name == 'adadelta':
        if learning_rate is None: learning_rate = 0.001
        if rho is None: rho = 0.95
        if epsilon is None: epsilon = 1e-7
        optimizer = Adadelta(learning_rate=learning_rate,
                             rho=rho,
                             epsilon=epsilon)
    elif optimizer_name == 'adagrad':
        if learning_rate is None: learning_rate = 0.001
        if initial_accumulator_value is None: initial_accumulator_value = 0.1
        if epsilon is None: epsilon = 1e-7
        optimizer = Adagrad(learning_rate=learning_rate,
                            initial_accumulator_value=initial_accumulator_value,
                            epsilon=epsilon)
    elif optimizer_name == 'adam':
        if learning_rate is not None: learning_rate = 0.001
        if beta_1 is None: beta_1 = 0.9
        if beta_2 is None: beta_2 = 0.999
        if epsilon is None: epsilon = 1e-7
        if amsgrad is None: amsgrad = False
        optimizer = Adam(learning_rate=learning_rate,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         epsilon=epsilon,
                         amsgrad=amsgrad)
    elif optimizer_name == 'adamax':
        if learning_rate is not None: learning_rate = 0.001
        if beta_1 is None: beta_1 = 0.9
        if beta_2 is None: beta_2 = 0.999
        if epsilon is None: epsilon = 1e-7
        optimizer = Adamax(learning_rate=learning_rate,
                           beta_1=beta_1,
                           beta_2=beta_2,
                           epsilon=epsilon)
    elif optimizer_name == 'ftrl':
        if learning_rate is not None: learning_rate = 0.001
        if learning_rate_power is None: learning_rate_power = -0.5
        if initial_accumulator_value is None: initial_accumulator_value = 0.1
        if l1_regularization_strength is None: l1_regularization_strength = 0.0
        if l2_regularization_strength is None: l2_regularization_strength = 0.0
        if l2_shrinkage_regularization_strength is None: l2_shrinkage_regularization_strength = 0.0
        if beta is None: beta = 0.0
        optimizer = Ftrl(learning_rate=learning_rate,
                         learning_rate_power=learning_rate_power,
                         initial_accumulator_value=initial_accumulator_value,
                         l1_regularization_strength=l1_regularization_strength,
                         l2_regularization_strength=l2_regularization_strength,
                         l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength,
                         beta=beta)
        
    elif optimizer_name == 'nadam':
        if learning_rate is not None: learning_rate = 0.001
        if beta_1 is None: beta_1 = 0.9
        if beta_2 is None: beta_2 = 0.999
        if epsilon is None: epsilon = 1e-7
        optimizer = Nadam(learning_rate=learning_rate,
                          beta_1=beta_1,
                          beta_2=beta_2,
                          epsilon=epsilon)
    elif optimizer_name == 'rmsprop':
        if learning_rate is not None: learning_rate = 0.001
        if rho is None: rho = 0.9
        if momentum is None: momentum = 0.0
        if epsilon is None: epsilon = 1e-7
        if centered is None: centered = False
        optimizer = RMSprop(learning_rate=learning_rate,
                            rho=rho,
                            momentum=momentum,
                            epsilon=epsilon,
                            centered=centered)
    elif optimizer_name == 'sgd':
        if learning_rate is not None: learning_rate = 0.001
        if momentum is None: momentum = 0.0
        if nesterov is None: nesterov = False
        optimizer = SGD(learning_rate=learning_rate,
                        momentum=momentum,
                        nesterov=nesterov)
    else:
        # This is the default value for the Tensorflow keras compile
        # function optimizer argument
        optimizer = 'rmsprop' 

    return optimizer

def _handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4


def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv3D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv', padding='same')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv3D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv3D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv', padding='same')(x)
    x = AveragePooling3D(1, strides=(2, 2, 2), name=name + '_pool', padding='same')(x)
    return x


# 组合模型
class DensenetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        print('original input shape:', input_shape)
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        print('original input shape:', input_shape)
        # orignal input shape: 1,7,7,200

        # if K.image_data_format() == 'channels_last':
        #     input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
        # print('change input shape:', input_shape)

        # 张量流输入
        input = Input(shape=input_shape)

        # 3D Convolution and pooling
        conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', kernel_initializer='he_normal')(
            input)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(conv1)

        # Dense Block1
        x = dense_block(pool1, 6, name='conv1')
        x = transition_block(x, 0.5, name='pool1')
        x = dense_block(x, 6, name='conv2')
        x = transition_block(x, 0.5, name='pool2')
        x = dense_block(x, 6, name='conv3')
        print(x.shape)
        x = GlobalAveragePooling3D(name='avg_pool')(x)
        print(x.shape)
        # x = Dense(16, activation='softmax')(x)

        # 输入分类器
        # Classifier block
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(x)

        model = Model(inputs=input, outputs=dense, name='3D-DenseNet')
        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        # (1,7,7,200),16
        return DensenetBuilder.build(input_shape, num_outputs)

class CNN3DBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        print('original input shape:', input_shape)
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        print('original input shape:', input_shape)
        # orignal input shape: 1,7,7,200

        # if K.image_data_format() == 'channels_last':
        #     input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
        # print('change input shape:', input_shape)

        input = Input(shape=input_shape)

        # conv1 = Conv3D(filters=128, kernel_size=(3, 3, 20), strides=(1, 1, 5),
        #                kernel_regularizer=regularizers.l2(0.01))(input)
        # act1 = Activation('relu')(conv1)
        # pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(act1)

        # conv2 = Conv3D(filters=192, kernel_size=(2, 2, 3), strides=(1, 1, 2),
        #                kernel_regularizer=regularizers.l2(0.01))(pool1)
        # act2 = Activation('relu')(conv2)
        # drop1 = Dropout(0.5)(act2)
        # pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(drop1)

        # conv3 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 2), padding='same',
        #                kernel_regularizer=regularizers.l2(0.01))(pool2)
        # act3 = Activation('relu')(conv3)
        # drop2 = Dropout(0.5)(act3)

        # flatten1 = Flatten()(drop2)
        # fc1 = Dense(200, kernel_regularizer=regularizers.l2(0.01))(flatten1)
        # act3 = Activation('relu')(fc1)

        conv1 = Conv3D(filters=32, kernel_size=(3, 3, 20), strides=(1, 1, 5), padding='same',
                       kernel_regularizer=regularizers.l2(0.01))(input)
        act1 = Activation('relu')(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(act1)

        conv2 = Conv3D(filters=64, kernel_size=(2, 2, 3), strides=(1, 1, 2), padding='same',
                       kernel_regularizer=regularizers.l2(0.01))(pool1)
        act2 = Activation('relu')(conv2)
        drop1 = Dropout(0.5)(act2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(drop1)

        conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 2), padding='same',
                       kernel_regularizer=regularizers.l2(0.01))(pool2)
        act3 = Activation('relu')(conv3)
        drop2 = Dropout(0.5)(act3)

        flatten1 = Flatten()(drop2)
        fc1 = Dense(num_outputs*2, kernel_regularizer=regularizers.l2(0.01))(flatten1)
        act3 = Activation('relu')(fc1)


        # Classifier block
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(act3)

        model = Model(inputs=input, outputs=dense, name='3D-CNN')
        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        # (1,7,7,200),16
        return CNN3DBuilder.build(input_shape, num_outputs)


def densenet_model(img_rows, img_cols, img_channels, nb_classes):

    model = DensenetBuilder.build_resnet_8(
        (1, img_rows, img_cols, img_channels), nb_classes)

    return model

def cnn_3d_model(img_rows, img_cols, img_channels, nb_classes):

    model = CNN3DBuilder.build_resnet_8(
        (1, img_rows, img_cols, img_channels), nb_classes)

    return model

def baseline_cnn_model(img_rows, img_cols, img_channels, 
                       patch_size, nb_filters, nb_classes):
    """
    Generates baseline CNN model for classifying HSI dataset.

    Parameters
    ----------
    img_rows : int
        Number of rows in neighborhood patch.
    img_cols : int
        Number of columns in neighborhood patch.
    img_channels : int
        Number of spectral bands.
    nb_classes : int
        Number of label categories.
    lr : float
        Learning rate for the model
    momentum : float
        Momentum value for optimizer

    Returns
    -------
    model : Model
        A keras API model of the constructed ML network.
    """

    model_input = Input(shape=(1, img_rows, img_cols, img_channels))
    conv_layer = Conv3D(nb_filters, (patch_size, patch_size, img_channels), 
                        strides=(1, 1, 1),name='3d_convolution_layer', padding='same',
                        kernel_regularizer=regularizers.l2(0.01))(model_input)
    activation_layer = Activation('relu', name='activation_layer')(conv_layer)
    max_pool_layer = MaxPooling3D(pool_size=(2, 2, 2), name='3d_max_pooling_layer', padding='same')(activation_layer)
    flatten_layer = Flatten(name='flatten_layer')(max_pool_layer)
    dense_layer = Dense(units=nb_classes, name='dense_layer')(flatten_layer)
    classifier_layer = Activation('softmax', name='classifier_layer')(dense_layer)

    model = Model(model_input, classifier_layer, name='baseline_cnn_model')

    return model