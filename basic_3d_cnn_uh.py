#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test harness module for 3D-DenseNet for University of Houston 2018
"""

### Built-in Imports ###
import argparse
import os
import time

### Other Library Imports ###
import numpy as np
from numpy.core.numeric import full_like
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import scipy.io as sio
from six import assertRaisesRegex
from sklearn import metrics, preprocessing

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv3D, Dense, Dropout, Flatten, Input, MaxPooling2D, MaxPooling3D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp

# import tensorflow.keras.callbacks as kcallbacks



### Local Imports ###

# N/A

### Environment Setup ###
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### Global Constants ###

# List of classes where the index is the value of the pixel in the
# ground truth image
CLASS_LIST = [
    'Undefined',
    'Healthy grass',
    'Stressed grass',
    'Artificial turf',
    'Evergreen trees',
    'Deciduous trees',
    'Bare earth',
    'Water',
    'Residential buildings',
    'Non-residential buildings',
    'Roads',
    'Sidewalks',
    'Crosswalks',
    'Major thoroughfares',
    'Highways',
    'Railways',
    'Paved parking lots',
    'Unpaved parking lots',
    'Cars',
    'Trains',
    'Stadium seats',
]

### Global Variables ###
verbose = False
debug = False
show_plots = False

### Definitions ###

def print_v(str):
    """
    Prints a string if the verbose flag is true.

    Parameters
    ----------
    str : str
        A string to print if verbosity is on.
    """
    if verbose:
        print(str)

def print_d(str):
    """
    Prints a string if the debug flag is true.

    Parameters
    ----------
    str : str
        A string to print if debug is on.
    """
    if debug:
        print(str)

def load_hs_images(file_path, hs_gsd, gt_gsd, num_tile_rows, num_tile_columns):

    # Initialize list for tiles
    hs_tiles = []
    
    # Set the factor of GSD resampling 1/factor for 1m to 0.5m
    resample_factor = hs_gsd / gt_gsd

    # Open the training HSI Envi file as src
    with rasterio.open(file_path, format='ENVI') as src:
        # Get the size of the tile windows
        tile_width = src.width / num_tile_columns
        tile_height = src.height / num_tile_rows

        # Read in the image data for each image tile
        for tile_row in range(0, num_tile_rows):
            hs_tiles.append([])
            for tile_column in range(0, num_tile_columns):

                # Set the tile window to read from the image
                window = Window(tile_width * tile_column,  
                                tile_height * tile_row, 
                                tile_width, tile_height)

                # Set the shape of the resampled tile
                out_shape=(src.count,
                           int(tile_height * resample_factor), 
                           int(tile_width * resample_factor))

                # Read the tile window from the image, resample it to
                # the appropriate GSD, arrange the numpy array to be
                # (rows, cols, bands), and remove unused bands
                tile = np.moveaxis(src.read(
                    window = window, 
                    out_shape=out_shape, 
                    resampling=Resampling.nearest), 0, -1)[:,:,:-2]
                
                # Copy the tile to the tiles array
                hs_tiles[tile_row].append(np.copy(tile))
    
    return np.stack(hs_tiles)

def load_ground_truth(train_path, test_path, 
                      num_tile_rows, num_tile_columns,
                      training_tile_list, training_offsets):
    # Initialize list for tiles
    gt_tiles = []

    with rasterio.open(train_path) as train_src, \
         rasterio.open(test_path) as test_src:

        # Get the size of the tile windows (use full size test image)
        tile_width = test_src.width / num_tile_columns
        tile_height = test_src.height / num_tile_rows

            # Read in the image data for each image tile
        for tile_row in range(0, num_tile_rows):
            gt_tiles.append([])
            for tile_column in range(0, num_tile_columns):

                # Check to see if current tile is one of the training
                # ground truth tiles
                if (tile_row, tile_column) in training_tile_list:

                    offset_row, offset_column = training_offsets[
                                                    (tile_row, tile_column)]

                    # Set the tile window to read from the image
                    window = Window(tile_width * offset_column ,  
                                    tile_height * offset_row, 
                                    tile_width, tile_height)

                    # Read the tile window from the image
                    tile = train_src.read(1, window = window)

                    # Copy the tile to the tiles array
                    gt_tiles[tile_row].append(np.copy(tile))
                else:
                    # Set the tile window to read from the image
                    window = Window(tile_width * tile_column,  
                                    tile_height * tile_row, 
                                    tile_width, tile_height)

                    # Read the tile window from the image
                    tile = test_src.read(1, window = window)

                    # Copy the tile to the tiles array
                    gt_tiles[tile_row].append(np.copy(tile))

    return np.stack(gt_tiles)

def merge_tiles(tiles):
    # Get number of tiles in nparray
    tile_rows, tile_cols = tiles.shape[:2]

    # Initialize empty list of image row values
    image_rows = []

    # Loop through each tile and stitch them together into single image
    for row in range(0, tile_rows):
        # Get first tile
        img_row = np.copy(tiles[row][0])

        # Loop through remaining tiles in current row
        for col in range(1, tile_cols):
            # Concatenate each subsequent tile in row to image row array
            img_row = np.concatenate((img_row, tiles[row][col]), axis=1)
        
        # Append image row to list of image rows
        image_rows.append(img_row)

    # Concatenate all image rows together to create single image
    merged_image = np.concatenate(image_rows, axis=0)

    return merged_image

def load_houston_2018_dataset():

    ### Constants ###

    HOUSTON_DATASET_PATH = '/content/drive/MyDrive/data/UH_2018_Dataset'
    TRAINING_GT_IMAGE_PATH = HOUSTON_DATASET_PATH + 'TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif'
    TESTING_GT_IMAGE_PATH = HOUSTON_DATASET_PATH + 'TestingGT/Test_Labels.tif'
    HSI_IMAGE_PATH = HOUSTON_DATASET_PATH + 'FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix'

    # Number of rows and columns in the overall dataset image such that 
    # parts of the dataset can be matched with the ground truth
    COLUMN_TILES = 7
    ROW_TILES = 2

    # List of tuples corresponding to the image tiles that match the ground
    # truth for the training set
    TRAINING_GT_TILES = ((1,1), (1,2), (1, 3), (1, 4))

    # Mapping of overall dataset training ground truth tile indices to the
    # tile indices of the actual ground truth image
    TRAINING_GT_TILE_OFFSETS = {
        (1,1) : (0,0),
        (1,2) : (0,1),
        (1,3) : (0,2),
        (1,4) : (0,3)
    }

    # GSD = Ground Sampling Distance
    HS_GSD = 1.0    # Hyperspectral image GSD in meters
    GT_GSD = 0.5    # Ground truth images GSD in meters

    # Number of hyperspectral band channels
    NUM_HS_BANDS = 48

    ### Load Data ###

    # Initialize variables
    tiled_dataset = {}

    # Load hyperspectral image data
    if os.path.exists('preprocessed_datasets/hs_tiles.npy'):
        with open('preprocessed_datasets/hs_tiles.npy', 'rb') as infile:
            print('Loading hyperspectral tile array...')
            tiled_dataset['hs'] = np.load(infile)
    else:
        print('Loading hyperspectral image...')
        tiled_dataset['hs'] = load_hs_images(HSI_IMAGE_PATH, 
            HS_GSD, GT_GSD, ROW_TILES, COLUMN_TILES)
        print('>>> Saving hyperspectral tile array...')
        with open('preprocessed_datasets/hs_tiles.npy', 'wb') as outfile:
            np.save(outfile, tiled_dataset['hs'])

    # Load ground truth data
    if os.path.exists('preprocessed_datasets/gt_tiles.npy'):
        with open('preprocessed_datasets/gt_tiles.npy', 'rb') as infile:
            print('Loading ground truth tile array...')
            tiled_dataset['gt'] = np.load(infile)
    else:
        print('Loading ground truth images...')
        tiled_dataset['gt'] = load_ground_truth(
            TRAINING_GT_IMAGE_PATH, TESTING_GT_IMAGE_PATH,
            ROW_TILES, COLUMN_TILES,
            TRAINING_GT_TILES, TRAINING_GT_TILE_OFFSETS)
        print('>>> Saving ground truth tile array...')
        with open('preprocessed_datasets/gt_tiles.npy', 'wb') as outfile:
            np.save(outfile, tiled_dataset['gt'])

    full_image_dataset = {
        'hs': merge_tiles(tiled_dataset['hs']),
        'gt': merge_tiles(tiled_dataset['gt']),
    }

    num_tile_rows, num_tile_cols, rows_per_tile, cols_per_tile, bands = tiled_dataset['hs'].shape

    train_indices = []
    test_indices = []

    for tr in range(num_tile_rows):
        for tc in range(num_tile_cols):
            for r in range(rows_per_tile):
                for c in range(cols_per_tile):
                    x = tc*cols_per_tile + c
                    y = tr*rows_per_tile + r

                    if tiled_dataset['gt'][tr][tc][r][c] != 0:
                        if (tr, tc) in TRAINING_GT_TILES:
                            train_indices.append(np.array((x, y)))
                        else:
                            test_indices.append(np.array((x, y)))


    return full_image_dataset['hs'], full_image_dataset['gt'], train_indices, test_indices

def create_model(input_shape, num_classes, learning_rate):
    """
    Creates 3D-CNN model.
    """

    # Create the model
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(64, 5, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss=categorical_crossentropy,
                  optimizer=RMSProp(learning_rate=learning_rate),
                  metrics=['accuracy'])

def run_model(model, X, y, 
              train_indices, test_indices, 
              batch_size, epochs, 
              validation_split):
    """
    Runs the ML model.
    """

    

def argument_parser():
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
    parser.add_argument('--batch-size', type=int, default=8,
            help='Number of samples that will propagate through the model \
                  at a single time')
    parser.add_argument('--epochs', type=int, default=1,
            help='Number of complete passes through dataset')
    parser.add_argument('--learning-rate', type=float, default=0.001,
            help='Percentage of training set that will be used for training')
    parser.add_argument('--validation-split', type=float, default=0.2,
            help='Percentage of training set that will be used for validation')

    return parser


if __name__ == "__main__":
    # Set up parser
    parser = argument_parser()
    args = parser.parse_args()

    # Get command line arguments
    show_plots = args.show_plots
    verbose = args.verbose
    debug = args.debug
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    validation_split = args.validation_split

    # delete unneeded variables
    del parser
    del args

    # Get dataset and training/testing indices
    X, y, train_indices, test_indices = load_houston_2018_dataset()
