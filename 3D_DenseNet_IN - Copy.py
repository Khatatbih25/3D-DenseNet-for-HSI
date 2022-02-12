#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test harness module for 3D-DenseNet for Indian Pines
"""

### Built-in Imports ###
import argparse
import collections
import os
import time

### Other Library Imports ###
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn import metrics, preprocessing
# from sklearn.decomposition import PCA
import tensorflow.keras.callbacks as kcallbacks
# from tensorflow.keras.layers import Activation, BatchNormalization, \
#         Convolution2D, Conv3D, Dense, Dropout, Flatten, Input, MaxPooling2D, \
#         MaxPooling3D, ZeroPadding3D
# from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop, SGD
# from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical


### Local Imports ###
from Utils import averageAccuracy, cnn_3D_IN, densenet_IN, \
        densenet_IN_no_bottleneck_layer, doPCA, modelStatsRecord, \
        zeroPadding

### Environment Setup ###
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### Constants ###

INPUT_DIMENSION_CONV = 200  # number of spectral bands
INPUT_DIMENSION = 200       # number of spectral bands

# The total split is 2:1:7 for train:validation:test
TOTAL_SIZE = 10249  # total number of samples across all classes
VAL_SIZE = 1025     # total number of samples in the validation dataset
TRAIN_SIZE = 2055   # total number of samples in the training dataset
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE # total number of samples in test set
VALIDATION_SPLIT = 0.8  # 20% for training and 80% for validation and testing

# Spatial context size (number of neighbours in each spatial direction)
PATCH_LENGTH = 5  

ITER = 1        # number of iterations to run this model
CATEGORY = 16   # number of classification categories in dataset

### Global Variables ###

# Load matlab data for Indian Pines dataset
mat_data = sio.loadmat('datasets/Indian_pines_corrected.mat')

# Get Indian Pines dataset array
data_IN = mat_data['indian_pines_corrected']

# Load the matlab data for Indian Pines ground truth
mat_gt = sio.loadmat('datasets/Indian_pines_gt.mat')

# Get Indian Pines ground truth array
gt_IN = mat_gt['indian_pines_gt']

new_gt_IN = gt_IN   # copy of ground truth array data
batch_size = 8      # number of samples to put through model in one shot
nb_classes = 16     # number of classification classes
nb_epoch = 15       # number of epochs to run model for
img_rows, img_cols = 11, 11     # Neighboring pixel block size
img_channels = 200  # number of spectral bands

# Number of epochs with no improvement after which training will be
# stopped
patience = 200

# Take the input data and reshape it from a 3-D array into a 2-D array
# by taking the product of the first two dimensions as the new first
# dimension and the product of the remaining dimensions (should be just
# one) as the second dimension
data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))

# Independently standardize each feature, center it, and scale each
# feature to the unit variance
data = preprocessing.scale(data)

# Reshape the ground truth to be only one dimension consisting of the
# product of the first two dimensions
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

# Create a nd array copy of the dataset with its first three dimensions
data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])

# Create a copy of the copy of the dataset
whole_data = data_

# Create an nd array copy of the dataset with padding at PATCH_LENGTH
# distance around the image
padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)

# Create zeroed out numpy arrays with dimensions 
# (# training samples, spatial-sample size, spatial-sample size, # bands)
# and
# (# testing samples, spatial-sample size, spatial-sample size, # bands)
train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))

# Initialize statistics lists
KAPPA_3D_DenseNet = []
OA_3D_DenseNet = []
AA_3D_DenseNet = []
TRAINING_TIME_3D_DenseNet = []
TESTING_TIME_3D_DenseNet = []
ELEMENT_ACC_3D_DenseNet = np.zeros((ITER, CATEGORY))

# A list of random number generator seeds where the seed at each index
# corresponds to the seed to use at that number iteration
seeds = [1334]

# Print variables for verification
print(f'data_IN.shape={data_IN.shape}')
print(f'gt_IN.shape={gt_IN.shape}')
print(f'data_IN.shape[:2]={data_IN.shape[:2]}')
print(f'np.prod(data_IN.shape[:2])={np.prod(data_IN.shape[:2])}')
print(f'data_IN.shape[2:]={data_IN.shape[2:]}')
print(f'np.prod(data_IN.shape[2:])={np.prod(data_IN.shape[2:])}')
print(f'np.prod(new_gt_IN.shape[:2])={np.prod(new_gt_IN.shape[:2])}')
print(f'data.shape={data.shape}')
print(f'padded_data.shape={padded_data.shape}')
print(f'train_data.shape={train_data.shape}')
print(f'test_data.shape={test_data.shape}')


### Definitions ###

def indexToAssignment(indices, Row, Col, pad_length):
    """
    Takes a list of indices to samples in the dataset and creates a new
    list of row-column index pairs.

    Parameters
    ----------
    indices : list of int
        A list of indices to the sample points on the dataset.
    Row : int
        The number of rows in the dataset.
    Col : int
        The number of columns in the dataset.
    pad_length : int
        The number of neighbors of the sample in each spatial direction.
    
    Returns
    -------
    new_assign : dictionary of lists of int
        A new list of row-column sample indicies.
    """

    # Initialize assignment dictionary
    new_assign = {}

    # Loop through the enumeration of the indices
    for counter, value in enumerate(indices):
        
        assign_0 = value // Col + pad_length    # Row assignment
        assign_1 = value % Col + pad_length     # Column assignment
        new_assign[counter] = [assign_0, assign_1] # Assign row-col pair
    
    return new_assign

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    """
    Selects the patch of neighbors for a particular sample point.

    Parameters
    ----------
    matrix : zero padded nparray
        The dataset from which to select the neighborhood patch.
    pos_row : int
        Row index of sample to find neighborhood of.
    pos_col : int
        Column index of sample to find neighborhood of.
    ex_len : int
        The number of neighbors in each spatial direction.

    Returns
    -------
    selected_patch : nparray
        The (ex_len*2+1) by (ex_len*2+1) matrix of samples in the
        (pos_row, pos_col) sample neighborhood.
    """
    # Narrow down the data matrix to the rows that are in the sample's
    # neighborhood
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    
    # Of the set of rows that are in the neighborhood, select the set
    # of columns in the neighborhood
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    
    return selected_patch


def sampling(proportionVal, groundTruth):
    """
    Divides the dataset into training and testing datasets by randomly
    sampling each class and separating the samples by validation split.

    Parameters
    ----------
    proportionVal : float
        The 0.0 < 'proportionVal' < 1.0 proportion of the entire dataset
        that will be used for validation/test set.
    groundTruth : nparray of int
        The dataset of ground truth classes.

    Returns
    -------
    train_indices : list of int
        A list of whole dataset indices that will be used for the
        training dataset.
    test_indices : list of int
        A list of whole dataset indices that will be used for the
        testing/validation dataset.
    """

    # Initialize label - sample dictionaries
    labels_loc = {}
    train = {}
    test = {}
    
    # Get the number of classes in the ground truth
    m = max(groundTruth)
    print(m)
    
    # Get a random sampling of each class for the training and testing
    # sets
    for i in range(m):
        # Get indicies of samples that belong to class i
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        
        # Shuffle the indicies 'randomly' (repeatable due to random seed)
        np.random.shuffle(indices)

        # Save the locations of all the matching samples for current
        # label
        labels_loc[i] = indices

        # Get the number of samples dedicated to the training set vs.
        # the testing set
        nb_val = int(proportionVal * len(indices))

        # Set (1-proportionVal) fraction of samples for this label to
        # the training set
        train[i] = indices[:-nb_val]

        # Set proportionVal fraction of samples for this label to
        # the testing/validation set
        test[i] = indices[-nb_val:]
    
    # Initialize lists for training and testing point indicies
    train_indices = []
    test_indices = []

    # Copy training and testing sample indicies to their respective list
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]

    # Shuffle the order of the sample indicies in the indices lists
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Print number of testing and training samples
    print(len(test_indices))
    print(len(train_indices))

    return train_indices, test_indices


def model_DenseNet():
    """
    Generates 3-D DenseNet model for classifying HSI dataset.

    Returns
    -------
    model_dense : Model
        A keras API model of the 3D DenseNet.
    """

    # Build DenseNet model with kernel (depth (?), rows, cols, bands) on
    # a number of classes
    model_dense = densenet_IN.ResnetBuilder.build_resnet_8(
        (1, img_rows, img_cols, img_channels), nb_classes)

    # Create RMSprop optimizer
    RMS = RMSprop(lr=0.0003)

    # Compile DenseNet model
    model_dense.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])

    return model_dense

def run_3d_densenet_in():
    """
    Runs the 3D-DenseNet for the Indian Pines dataset.
    """

    # Run 3-D DenseNet for ITER iterations
    for index_iter in range(ITER):
        print(f'>>> Iteration #{index_iter + 1}')

        # Path for saving the best validated model at the model
        # checkpoint
        best_weights_DenseNet_path = 'training_results/indian_pines/Indian_best_3D_DenseNet_1' + str(
            index_iter + 1) + '.hdf5'

        # Initialize random seed for sampling function
        np.random.seed(seeds[index_iter])

        # Randomly sample each class into the training set and the
        # testing/validation set
        train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)

        # Create training set class vector
        y_train = gt[train_indices] - 1

        # Convert training set class vector into binary class matrix 
        # for one-hot encoding
        y_train = to_categorical(np.asarray(y_train))

        # Create testing set class vector
        y_test = gt[test_indices] - 1

        # Convert testing set class vector into binary class matrix 
        # for one-hot encoding
        y_test = to_categorical(np.asarray(y_test))

        # Get row-column pair assignments for training set
        train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
        
        # Loop through row-column training assignments to get the set of
        # neighborhood patches for each training sample
        for i in range(len(train_assign)):
            train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

        # Get row-column pair assignments for testing set
        test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
        
        # Loop through row-column testing assignments to get the set of
        # neighborhood patches for each testing sample
        for i in range(len(test_assign)):
            test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

        # 拿到了新的数据集进行reshpae之后，数据处理就结束了
        x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
        x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)

        # 在测试数据集上进行验证和测试的划分
        x_val = x_test_all[-VAL_SIZE:]
        y_val = y_test[-VAL_SIZE:]

        x_test = x_test_all[:-VAL_SIZE]
        y_test = y_test[:-VAL_SIZE]

        ############################################################################################################
        # 在这里对所使用模型进行设计，这代码不复用真实可耻
        model_densenet = model_DenseNet()

        # monitor：监视数据接口，此处是val_loss,patience是在多少步可以容忍没有提高变化
        earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
        # 用户每次epoch最后都会保存模型，如果save_best_only=True,那么最近验证误差最后的数据将会被保存下来
        saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights_DenseNet_path, monitor='val_loss', verbose=1,
                                                    save_best_only=True,
                                                    mode='auto')

        # 训练和验证
        tic6 = time.process_time()
        print(x_train.shape, x_test.shape)
        # (2055,7,7,200)  (7169,7,7,200)

        # history_3d_densenet = model_densenet.fit(
        #     x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1), y_train,
        #     validation_data=(x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3], 1), y_val),
        #     batch_size=batch_size,
        #     epochs=nb_epoch, shuffle=True, callbacks=[earlyStopping6, saveBestModel6])

        history_3d_densenet = model_densenet.fit(
            x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2], x_train.shape[3]), y_train,
            validation_data=(x_val.reshape(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2], x_val.shape[3]), y_val),
            batch_size=batch_size,
            epochs=nb_epoch, shuffle=True, callbacks=[earlyStopping6, saveBestModel6])
        toc6 = time.process_time()

        # 测试
        tic7 = time.process_time()
        # loss_and_metrics = model_densenet.evaluate(
        #     x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1), y_test,
        #     batch_size=batch_size)
        loss_and_metrics = model_densenet.evaluate(
            x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2], x_test.shape[3]), y_test,
            batch_size=batch_size)
        toc7 = time.process_time()

        print('3D DenseNet Time: ', toc6 - tic6)
        print('3D DenseNet Test time:', toc7 - tic7)

        print('3D DenseNet Test score:', loss_and_metrics[0])
        print('3D DenseNet Test accuracy:', loss_and_metrics[1])

        # print(history_3d_densenet.history.keys())
        # dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

        # 预测
        # pred_test = model_densenet.predict(
        #     x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)).argmax(axis=1)
        pred_test = model_densenet.predict(
            x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2], x_test.shape[3])).argmax(axis=1)
        # 跟踪值出现的次数
        collections.Counter(pred_test)

        gt_test = gt[test_indices] - 1
        # print(len(gt_test))
        # 8194
        # 这是测试集，验证和测试还没有分开
        overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
        confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
        each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
        kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])
        KAPPA_3D_DenseNet.append(kappa)
        OA_3D_DenseNet.append(overall_acc)
        AA_3D_DenseNet.append(average_acc)
        TRAINING_TIME_3D_DenseNet.append(toc6 - tic6)
        TESTING_TIME_3D_DenseNet.append(toc7 - tic7)
        ELEMENT_ACC_3D_DenseNet[index_iter, :] = each_acc

        print("3D DenseNet finished.")
        print("# %d Iteration" % (index_iter + 1))

    # 自定义输出类
    modelStatsRecord.outputStats(KAPPA_3D_DenseNet, OA_3D_DenseNet, AA_3D_DenseNet, ELEMENT_ACC_3D_DenseNet,
                                TRAINING_TIME_3D_DenseNet, TESTING_TIME_3D_DenseNet,
                                history_3d_densenet, loss_and_metrics, CATEGORY,
                                'training_results/indian_pines/IN_train_3D_10_.txt',
                                'training_results/indian_pines/IN_train_3D_element_10_.txt')

def test_harness_parser():
    """
    Sets up the parser for command-line flags for the test harness 
    script.

    Return
    ------
    argparse.ArgumentParser
        An ArgumentParser object configured with the test_harness.py
        command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str,
            help='Path to the dataset to run test harness on.')
    parser.add_argument('--batch-size', type=int, default=8, 
            help='Number of samples that will propagate through the model at a time.')
    parser.add_argument('--number-of-classes', type=int, default=16,
            help='Number of classes to classify.')
    parser.add_argument('--epochs', type=int, default=15,
            help='Number of complete passes through dataset')
    parser.add_argument('--validation-split', type=float, default=0.8,
            help='Percentage of training set that will be used for training')
    parser.add_argument('--image-rows', type=int, default=11,
            help='Number of image rows.')
    parser.add_argument('--image-cols', type=int, default=11,
            help='Number of image columns.')
    parser.add_argument('--best-weights-path', type=str, default='./',
            help='Path to location for storing best model weights.')
    parser.add_argument('--results-path', type=str, default='./',
            help='Path to location for experiment results.')
    parser.add_argument('--test-name', type=str, default='experiment',
            help='Name to use as the prefix for all output files')
    parser.add_argument('--verbose', action='store_true',
            help='Sets output to be more verbose.')

    return parser


### Main ###

if __name__ == "__main__":
    # Arguments
    # parser = test_harness_parser()
    # args = parser.parse_args()

    # dataset_path = args.dataset_path
    # outfile_prefix = args.test_name
    # batch_size = args.batch_size
    # num_classes = args.number_of_classes
    # epochs = args.epochs
    # validation_split = args.validation_split
    # image_rows = args.image_rows
    # image_cols = args.image_cols
    # verbose = args.verbose

    run_3d_densenet_in()
