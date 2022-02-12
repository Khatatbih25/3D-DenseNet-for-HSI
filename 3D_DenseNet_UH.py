#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test harness module for 3D-DenseNet for University of Houston 2018
"""

### Built-in Imports ###
import argparse
import collections
import os
import time

### Other Library Imports ###
import numpy as np
from numpy.core.numeric import full_like
import scipy.io as sio
from sklearn import metrics, preprocessing
import tensorflow.keras.callbacks as kcallbacks
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical


### Local Imports ###
from grss_dfc_2018_uh import NUMBER_OF_UH_2018_CLASSES, UH_2018_Dataset
from Utils import averageAccuracy, densenet_IN, modelStatsRecord, zeroPadding
import utilities

### Environment Setup ###
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### Global Constants ###

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


def model_DenseNet(img_rows, img_cols, img_channels, nb_classes):
    """
    Generates 3-D DenseNet model for classifying HSI dataset.

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

def run_3d_densenet_uh():
    """
    Runs the 3D-DenseNet for the University of Houston dataset.
    """

    dataset = UH_2018_Dataset()
    dataset.load_full_gt_image()
    dataset.load_full_hs_image()

    train_indices, test_indices = dataset.get_train_test_split(flatten=True)

    data = dataset.hs_image
    gt = dataset.gt_image

    # tile_list=((0,2),(1,2),(0,3),(1,3))

    # data = dataset.merge_tiles(dataset.load_hs_image_tiles(tile_list=tile_list), num_rows=2, num_cols=2)[...,::8]
    # gt = dataset.merge_tiles(dataset.load_gt_image_tiles(tile_list=tile_list), num_rows=2, num_cols=2)

    # # print(f'gt.shape={gt.shape}')

    # image_height, image_width = gt.shape
    # tile_width = int(image_width / 2)
    # tile_height = int(image_height / 2)

    # train_indices = []
    # test_indices = []

    # for col in range(image_width):
    #     for row in range(image_height):
    #         flat_index = row*image_width + col
    #         if gt[row][col] > 0:
    #             if row > tile_height:
    #                 train_indices.append(flat_index)
    #             else:
    #                 test_indices.append(flat_index)

    # train_indices = np.asarray(train_indices)
    # test_indices = np.asarray(test_indices)

    # train_indices = np.asarray(
    #     [row*image_width+col for row,col in dataset.get_tile_indices((1,2),row_offset=1)] +
    #     [row*image_width+col for row,col in dataset.get_tile_indices((1,3),row_offset=1, col_offset=1)])
    
    # test_indices = np.asarray(
    #     [row*image_width+col for row,col in dataset.get_tile_indices((0,2))] +
    #     [row*image_width+col for row,col in dataset.get_tile_indices((0,3),col_offset=1)])

    TRAIN_VAL_SPLIT = 0.2   # Use 20% of training samples for validation

    ### Set Constants ###
    INPUT_DIMENSION_CONV = data.shape[-1]

    dataset.clear_all_images()

    TOTAL_SIZE = len(train_indices) + len(test_indices) # total number of samples across all classes
    VAL_SIZE = int(len(train_indices) * TRAIN_VAL_SPLIT)     # total number of samples in the validation dataset
    TRAIN_SIZE = len(train_indices)          # total number of samples in the training dataset
    TEST_SIZE = len(test_indices) # total number of samples in test set
    VALIDATION_SPLIT = 0.8  # 20% for training and 80% for validation and testing

    # Spatial context size (number of neighbours in each spatial direction)
    PATCH_LENGTH = 1

    ITER = 1        # number of iterations to run this model
    CATEGORY = dataset.gt_num_classes

    ### Set Variables ###

    # Get Indian Pines dataset array
    data_UH = data

    # Get Indian Pines ground truth array
    gt_UH = gt

    new_gt_UH = gt_UH   # copy of ground truth array data
    batch_size = 128     # number of samples to put through model in one shot
    nb_epoch = 1        # number of epochs to run model for

    img_rows = PATCH_LENGTH * 2 + 1 # number of rows in neighborhood
    img_cols = PATCH_LENGTH * 2 + 1 # number of cols in neighborhood
    bands = INPUT_DIMENSION_CONV    # number of spectral bands
    classes = CATEGORY              # number of label categories

    # Number of epochs with no improvement after which training will be
    # stopped
    patience = 5

    # Take the input data and reshape it from a 3-D array into a 2-D array
    # by taking the product of the first two dimensions as the new first
    # dimension and the product of the remaining dimensions (should be just
    # one) as the second dimension
    data = data_UH.reshape(np.prod(data_UH.shape[:2]), np.prod(data_UH.shape[2:]))

    # Independently standardize each feature, center it, and scale each
    # feature to the unit variance
    data = preprocessing.scale(data)

    # Print variables for verification
    print(f'data_UH.shape={data_UH.shape}')
    print(f'gt_UH.shape={gt_UH.shape}')
    print(f'data_UH.shape[:2]={data_UH.shape[:2]}')
    print(f'np.prod(data_UH.shape[:2])={np.prod(data_UH.shape[:2])}')
    print(f'data_UH.shape[2:]={data_UH.shape[2:]}')
    print(f'np.prod(data_UH.shape[2:])={np.prod(data_UH.shape[2:])}')
    print(f'np.prod(new_gt_UH.shape[:2])={np.prod(new_gt_UH.shape[:2])}')
    print(f'data.shape={data.shape}')

    # Reshape the ground truth to be only one dimension consisting of the
    # product of the first two dimensions
    gt = new_gt_UH.reshape(np.prod(new_gt_UH.shape[:2]), )

    # Create a nd array copy of the dataset with its first three dimensions
    whole_data = data.reshape(data_UH.shape[0], data_UH.shape[1], data_UH.shape[2])

    # Create an nd array copy of the dataset with padding at PATCH_LENGTH
    # distance around the image
    padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)

    # Adjust training and testing indices

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
    print(f'data_UH.shape={data_UH.shape}')
    print(f'gt_UH.shape={gt_UH.shape}')
    print(f'data_UH.shape[:2]={data_UH.shape[:2]}')
    print(f'np.prod(data_UH.shape[:2])={np.prod(data_UH.shape[:2])}')
    print(f'data_UH.shape[2:]={data_UH.shape[2:]}')
    print(f'np.prod(data_UH.shape[2:])={np.prod(data_UH.shape[2:])}')
    print(f'np.prod(new_gt_UH.shape[:2])={np.prod(new_gt_UH.shape[:2])}')
    print(f'data.shape={data.shape}')
    print(f'padded_data.shape={padded_data.shape}')
    print(f'train_data.shape={train_data.shape}')
    print(f'test_data.shape={test_data.shape}')

    # Run 3-D DenseNet for ITER iterations
    for index_iter in range(ITER):
        print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
        print(f'>>> Iteration #{index_iter + 1} >>>')

        # Path for saving the best validated model at the model
        # checkpoint
        best_weights_DenseNet_path = 'training_results/university_of_houston/UHouston_best_3D_DenseNet_1' + str(
            index_iter + 1) + '.hdf5'

        # Initialize random seed for sampling function
        np.random.seed(seeds[index_iter])

        print(f'unique train indices: {np.unique(gt[train_indices])}')

        # Create training set class vector
        y_train = gt[train_indices] - 1

        # Convert training set class vector into binary class matrix 
        # for one-hot encoding
        y_train = to_categorical(np.asarray(y_train), num_classes=NUMBER_OF_UH_2018_CLASSES)

        # Create testing set class vector
        y_test = gt[test_indices] - 1

        # Convert testing set class vector into binary class matrix 
        # for one-hot encoding
        y_test = to_categorical(np.asarray(y_test), num_classes=NUMBER_OF_UH_2018_CLASSES)

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

        # Shape training and testing dataset features sets to 
        # (#samples, rows, cols, bands)
        x_train_all = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
        x_test = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)

        # Break part of training dataset out into validation dataset
        x_val = x_train_all[-VAL_SIZE:]
        y_val = y_train[-VAL_SIZE:]

        # Remove validation dataset from training dataset
        x_train = x_train_all[:-VAL_SIZE]
        y_train = y_train[:-VAL_SIZE]

        ############################################################################################################
        # Model creation, training, and testing
        model_densenet = model_DenseNet(img_rows, img_cols, bands, classes)

        # Create callback to stop training early if metrics don't improve
        cb_early_stopping = kcallbacks.EarlyStopping(monitor='val_loss', 
            patience=patience, verbose=1, mode='auto')

        # Create callback to save model weights if the model performs
        # better than the previously trained models
        cb_save_best_model = kcallbacks.ModelCheckpoint(best_weights_DenseNet_path, 
            monitor='val_loss', verbose=1, save_best_only=True, mode='auto')


        # Record start time for model training
        model_train_start = time.process_time()
        
        print(f'x_train shape: {x_train.shape}')
        print(f'x_train.reshape: {x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2], x_train.shape[3]).shape}')
        print(f'y_train shape: {y_train.shape}')
        print(f'x_val shape: {x_val.shape}')
        print(f'x_val.reshape: {x_val.reshape(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2], x_val.shape[3]).shape}')
        print(f'y_val shape: {y_val.shape}')

        # Train the 3D-DenseNet
        history_3d_densenet = model_densenet.fit(
            x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2], x_train.shape[3]), y_train,
            validation_data=(x_val.reshape(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2], x_val.shape[3]), y_val),
            batch_size=batch_size,
            epochs=nb_epoch, shuffle=True, callbacks=[cb_early_stopping, cb_save_best_model])
        
        # Record end time for model training
        model_train_end = time.process_time()

        # Record start time for model evaluation
        model_test_start = time.process_time()

        # Evaluate the trained 3D-DenseNet
        loss_and_metrics = model_densenet.evaluate(
            x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2], x_test.shape[3]), y_test,
            batch_size=batch_size)

        # Record end time for model evaluation
        model_test_end = time.process_time()

        # Print time metrics
        print('3D DenseNet Time: ', model_train_end - model_train_start)
        print('3D DenseNet Test time:', model_test_end - model_test_start)

        # Print loss and accuracy metrics
        print('3D DenseNet Test score:', loss_and_metrics[0])
        print('3D DenseNet Test accuracy:', loss_and_metrics[1])

        # Get prediction values for test dataset
        pred_test = model_densenet.predict(
            x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2], x_test.shape[3])).argmax(axis=1)
        
        # Store the prediction label counts
        collections.Counter(pred_test)

        # Create test class vector
        gt_test = gt[test_indices] - 1
        
        # Get prediction accuracy metric
        overall_acc = metrics.accuracy_score(pred_test, gt_test)
        
        # Get prediction confusion matrix
        confusion_matrix = metrics.confusion_matrix(pred_test, gt_test)
        
        # Get individual class accuracy as well as average accuracy
        each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
        
        # Get Kappa metric from predictions
        kappa = metrics.cohen_kappa_score(pred_test, gt_test)
        
        # Append all metrics to their respective lists
        KAPPA_3D_DenseNet.append(kappa)
        OA_3D_DenseNet.append(overall_acc)
        AA_3D_DenseNet.append(average_acc)

        # Append training and testing times to their respective lists
        TRAINING_TIME_3D_DenseNet.append(model_train_end - model_train_start)
        TESTING_TIME_3D_DenseNet.append(model_test_end - model_test_start)
        
        # Save individual accuracies to iteration index in element
        # accuracy list
        ELEMENT_ACC_3D_DenseNet[index_iter, :] = each_acc

        print("3D DenseNet finished.")
        print(f'<<< Iteration #{index_iter + 1} <<<')
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    # Print out the overall training and testing results for the model
    # and save the results to a file
    modelStatsRecord.outputStats(KAPPA_3D_DenseNet, OA_3D_DenseNet, AA_3D_DenseNet, ELEMENT_ACC_3D_DenseNet,
                                TRAINING_TIME_3D_DenseNet, TESTING_TIME_3D_DenseNet,
                                history_3d_densenet, loss_and_metrics, CATEGORY,
                                'training_results/university_of_houston/UH_train_3D_10_.txt',
                                'training_results/university_of_houston/UH_train_3D_element_10_.txt')


def uh_3d_densenet_parser():
    """
    Sets up the parser for command-line flags for the test harness 
    script.

    Returns
    -------
    argparse.ArgumentParser
        An ArgumentParser object configured with the test_harness.py
        command-line arguments.
    """

    SCRIPT_DESCRIPTION = ('Test harness script for experimenting on the '
                      'University of Houston 2018 GRSS Data Fusion Contest '
                      'dataset with the 3D-DenseNet model for hyperspectral '
                      'images.')

    parser = argparse.ArgumentParser(SCRIPT_DESCRIPTION)
    parser.add_argument('--show-plots', action='store_true',
            help='Turns on figures and plot displays.')
    parser.add_argument('--verbose', action='store_true',
            help='Sets output to be more verbose.')
    parser.add_argument('--debug', action='store_true',
            help='Enables debug output.')

    return parser

### Main ###
if __name__ == "__main__":

    # Set up parser
    parser = uh_3d_densenet_parser()
    args = parser.parse_args()

    # Get command line arguments
    show_plots = args.show_plots
    verbose = args.verbose
    debug = args.debug

    # Run Model
    run_3d_densenet_uh()
