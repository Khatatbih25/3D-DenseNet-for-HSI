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
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import scipy.io as sio
from sklearn import metrics, preprocessing
import tensorflow.keras.callbacks as kcallbacks
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical


### Local Imports ###
from Utils import averageAccuracy, densenet_IN, modelStatsRecord, zeroPadding

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

# def LoadHouston2018Dataset():
#     """
#     Loads the GRSS DFC 2018 Contest Houston Dataset into training and
#     testing datasets
#     """

#     HOUSTON_DATASET_PATH = 'datasets/grss_dfc_2018/'
#     TRAINING_GT_IMAGE_PATH = HOUSTON_DATASET_PATH + 'TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif'
#     TESTING_GT_IMAGE_PATH = HOUSTON_DATASET_PATH + 'TestingGT/Test_Labels.tif'
#     HSI_IMAGE_PATH = HOUSTON_DATASET_PATH + 'FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix'
#     LIDAR_DSM_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/DSM_C12/UH17c_GEF051.tif'
#     LIDAR_DEM_3MSR_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/DEM_C123_3msr/UH17_GEG051.tif'
#     LIDAR_DEM_TLI_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/DEM_C123_TLI/UH17_GEG05.tif'
#     LIDAR_DEM_B_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/DEM+B_C123/UH17_GEM051.tif'
#     LIDAR_INTENSITY_1550NM_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/Intensity_C1/UH17_GI1F051.tif'
#     LIDAR_INTENSITY_1064NM_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/Intensity_C2/UH17_GI2F051.tif'
#     LIDAR_INTENSITY_532NM_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/Intensity_C3/UH17_GI3F051.tif'

#     # Initialize Variables
#     hs_data = None  # (Rows, Cols, Spectral Bands), value is intensity
#     gt_train = None # (Rows, Cols), value is class id
#     gt_test = None  # (Rows, Cols), value is class id 
#     lidar_dsm = None
#     lidar_dem_3msr = None
#     lidar_dem_tli = None
#     lidar_dem_b = None

#     # Open ground truth files
#     print_v('Loading Houston 2018 ground truth image data...')
#     try:
#         with rasterio.open(TRAINING_GT_IMAGE_PATH) as gt_train_src, \
#             rasterio.open(TESTING_GT_IMAGE_PATH) as gt_test_src:

#             gt_train = gt_train_src.read(1)
#             gt_test = gt_test_src.read(1)
#     except IOError as e:
#         print('Could not load Houston 2018 ground truth!')
#         print(e)
#     except Exception as e:
#         print('An error occured loading Houston 2018 ground truth!')
#         print(e)
#     else:
#         print_v('Houston 2018 ground truth image data successfully loaded!')
        
#     # Open the HSI Envi file 
#     print_v('Loading Houston 2018 HSI image data...')
#     try:
#         with rasterio.open(HSI_IMAGE_PATH, format='ENVI') as src:
#             # Don't read original image to print unless in verbose mode,
#             # as it is less efficient
#             if verbose:
#                 orig = np.moveaxis(src.read(), 0, -1)[:,:,:-2]
#                 print_v(f'Original HSI shape (before resampling): {orig.shape}')
            
#             # Get hyperspectral image data, resampled to 0.5m GSD
#             hs_data = np.moveaxis(src.read(out_shape=(
#                 src.count, gt_test.shape[0], gt_test.shape[1]), 
#                 resampling=Resampling.nearest), 0, -1)[:,:,:-2]
#     except IOError as e:
#         print('Could not load Houston 2018 HSI images!')
#         print(e)
#     except Exception as e:
#         print('An error occured loading Houston 2018 HSI images!')
#         print(e)
#     else:
#         print_v('Houston 2018 HSI image data successfully loaded!')


#     # Open lidar rasters
#     # print_v('Loading Houston 2018 LiDAR data...')
#     # try:
#     #     with rasterio.open(LIDAR_DSM_PATH) as lidar_dsm_src, \
#     #          rasterio.open(LIDAR_DEM_3MSR_PATH) as lidar_dem_3msr_src, \
#     #          rasterio.open(LIDAR_DEM_TLI_PATH) as lidar_dem_tli_src, \
#     #          rasterio.open(LIDAR_DEM_B_PATH) as lidar_dem_b_src:

#     #         # Read in Lidar Digital Surface Model Channels 1 & 2 image
#     #         # data and move band dimension from first to last
#     #         lidar_dsm = np.moveaxis(lidar_dsm_src.read(), 0, -1)

#     #         # Read in Lidar Digital Elevation Model 3MSR image data and 
#     #         # move band dimension from first to last
#     #         lidar_dem_3msr = np.moveaxis(lidar_dem_3msr_src.read(), 0, -1)

#     #         # Read in Lidar Digital Elevation Model TLI image data and 
#     #         # move band dimension from first to last
#     #         lidar_dem_tli = np.moveaxis(lidar_dem_tli_src.read(), 0, -1)

#     #         # Read in Lidar Digital Elevation Model B image data and 
#     #         # move band dimension from first to last
#     #         lidar_dem_b = np.moveaxis(lidar_dem_b_src.read(), 0, -1)
#     # except IOError as e:
#     #     print('Could not load Houston 2018 LiDAR data!')
#     #     print(e)
#     # except Exception as e:
#     #     print('An error occured loading Houston 2018 LiDAR data!')
#     #     print(e)
#     # else:
#     #     print_v('Houston 2018 LiDAR data successfully loaded!')


#     # If verbose, print out dataset information
#     print_v(f'gt_train shape: {gt_train.shape}')
#     print_v(f'gt_test shape:  {gt_test.shape}')
#     print_v(f'hsi shape:      {hs_data.shape}')
#     # print_v(f'lidar_dsm shape:      {lidar_dsm.shape}')
#     # print_v(f'lidar_dem_3msr shape: {lidar_dem_3msr.shape}')
#     # print_v(f'lidar_dem_tli shape:  {lidar_dem_tli.shape}')
#     # print_v(f'lidar_dem_b shape:    {lidar_dem_b.shape}')
    
#     return hs_data, gt_train, gt_test #, lidar_dsm, lidar_dem_3msr, lidar_dem_tli, lidar_dem_b

def load_vhr_images(file_paths, vhr_gsd, gt_gsd):

    # Initialize list for tiles
    vhr_tiles = []

    # Set the factor of GSD resampling 1/factor for 1m to 0.5m
    resample_factor = vhr_gsd / gt_gsd

    for row, tile_paths in enumerate(file_paths):
        vhr_tiles.append([])
        for column, tile_path in enumerate(tile_paths):
            with rasterio.open(tile_path) as src:
                # Set the shape of the resampled tile
                out_shape=(src.count,
                           int(src.height * resample_factor), 
                           int(src.width * resample_factor))

                # Read the tile image and resample it to
                # the appropriate GSD, arrange the numpy array to be
                # (rows, cols, bands)
                tile = np.moveaxis(src.read(
                    out_shape=out_shape, 
                    resampling=Resampling.nearest), 0, -1)
                
                # Copy the tile to the tiles array
                vhr_tiles[row].append(np.copy(tile))
    
    return np.stack(vhr_tiles)

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

def load_lidar_intensities(file_paths, num_tile_rows, num_tile_columns):
    # Initialize list for tiles
    lidar_intensity_tiles = []

    # Open the training HSI Envi file as src
    with rasterio.open(file_paths[0]) as c1_src, \
         rasterio.open(file_paths[1]) as c2_src, \
         rasterio.open(file_paths[2]) as c3_src:

        # Get the size of the tile windows (all should be same)
        tile_width = c1_src.width / num_tile_columns
        tile_height = c1_src.height / num_tile_rows

        # Read in the image data for each image tile
        for tile_row in range(0, num_tile_rows):
            lidar_intensity_tiles.append([])
            for tile_column in range(0, num_tile_columns):

                # Set the tile window to read from the image
                window = Window(tile_width * tile_column,  
                                tile_height * tile_row, 
                                tile_width, tile_height)

                # Read the tile window from the images
                c1_tile = c1_src.read(1, window = window)
                c2_tile = c2_src.read(1, window = window)
                c3_tile = c3_src.read(1, window = window)
                
                # Copy the tile to the tiles array
                lidar_intensity_tiles[tile_row].append(
                        np.dstack((c1_tile, c2_tile, c3_tile)))
    
    return np.stack(lidar_intensity_tiles)

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

def load_houston_dataset():

    ### Constants ###

    HOUSTON_DATASET_PATH = 'datasets/grss_dfc_2018/'
    TRAINING_GT_IMAGE_PATH = HOUSTON_DATASET_PATH + 'TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif'
    TESTING_GT_IMAGE_PATH = HOUSTON_DATASET_PATH + 'TestingGT/Test_Labels.tif'
    HSI_IMAGE_PATH = HOUSTON_DATASET_PATH + 'FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix'
    LIDAR_DSM_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/DSM_C12/UH17c_GEF051.tif'
    LIDAR_DEM_3MSR_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/DEM_C123_3msr/UH17_GEG051.tif'
    LIDAR_DEM_TLI_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/DEM_C123_TLI/UH17_GEG05.tif'
    LIDAR_DEM_B_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/DEM+B_C123/UH17_GEM051.tif'
    LIDAR_INTENSITY_1550NM_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/Intensity_C1/UH17_GI1F051.tif'
    LIDAR_INTENSITY_1064NM_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/Intensity_C2/UH17_GI2F051.tif'
    LIDAR_INTENSITY_532NM_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/Intensity_C3/UH17_GI3F051.tif'

    # Paths, in order of tile, for the very high resolution RGB image
    VHR_IMAGE_PATHS = [
        [
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_271460_3290290.tif',
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_272056_3290290.tif',
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_272652_3290290.tif',
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_273248_3290290.tif',
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_273844_3290290.tif',
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_274440_3290290.tif',
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_275036_3290290.tif',
        ],
        [
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_271460_3289689.tif',
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_272056_3289689.tif',
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_272652_3289689.tif',
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_273248_3289689.tif',
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_273844_3289689.tif',
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_274440_3289689.tif',
            HOUSTON_DATASET_PATH + 'Final RGB HR Imagery/UH_NAD83_275036_3289689.tif',
        ],
    ]

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
    VHR_GSD = 0.05  # Very high resolution image GSD in meters
    HS_GSD = 1.0    # Hyperspectral image GSD in meters
    LIDAR_GSD = 0.5 # LiDAR raster image GSD in meters
    GT_GSD = 0.5    # Ground truth images GSD in meters

    # A list of the wavelength values for each lidar band channel
    LIDAR_BAND_WAVELENGTHS = [
        '1550nm',
        '1064nm',
        '532nm'
    ]

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


    # Load lidar intensity data
    if os.path.exists('preprocessed_datasets/lidar_intensity_tiles.npy'):
        with open('preprocessed_datasets/lidar_intensity_tiles.npy', 'rb') as infile:
            print('Loading lidar intensity tile array...')
            tiled_dataset['lidar'] = np.load(infile)
    else:
        print('Loading lidar intensity images...')
        tiled_dataset['lidar'] = load_lidar_intensities(
                [LIDAR_INTENSITY_1064NM_PATH, 
                LIDAR_INTENSITY_1550NM_PATH, 
                LIDAR_INTENSITY_532NM_PATH], 
            ROW_TILES, COLUMN_TILES)
        print('>>> Saving lidar intensity tile array...')
        with open('preprocessed_datasets/lidar_intensity_tiles.npy', 'wb') as outfile:
            np.save(outfile, tiled_dataset['lidar'])


    # Load very high resolution rgb image data
    if os.path.exists('preprocessed_datasets/vhr_tiles.npy'):
        with open('preprocessed_datasets/vhr_tiles.npy', 'rb') as infile:
            print('Loading very high resolution rgb tile array...')
            tiled_dataset['vhr'] = np.load(infile)
    else:
        print('Loading very high resolution rgb images...')
        tiled_dataset['vhr'] = load_vhr_images(VHR_IMAGE_PATHS,VHR_GSD, GT_GSD)
        print('>>> Saving very high resolution rgb tile array...')
        with open('preprocessed_datasets/vhr_tiles.npy', 'wb') as outfile:
            np.save(outfile, tiled_dataset['vhr'])


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
        'lidar': merge_tiles(tiled_dataset['lidar']),
        'vhr': merge_tiles(tiled_dataset['vhr']),
        'gt': merge_tiles(tiled_dataset['gt']),
    }

    print(f"tiled_dataset['hs'].shape: {tiled_dataset['hs'].shape}")

    num_tile_rows, num_tile_cols, rows_per_tile, cols_per_tile, bands = tiled_dataset['hs'].shape

    train_indices = []
    test_indices = []

    for tr in range(num_tile_rows):
        for tc in range(num_tile_cols):
            for tile_y in range(rows_per_tile):
                for tile_x in range(cols_per_tile):
                    x = tc*cols_per_tile + tile_x
                    y = tr*rows_per_tile + tile_y

                    if tiled_dataset['gt'][tr][tc][tile_y][tile_x] != 0:
                        if (tr, tc) in TRAINING_GT_TILES:
                            train_indices.append(np.array((x, y)))
                        else:
                            test_indices.append(np.array((x, y)))

    # training_tiles = []
    # testing_tiles = []

    # for tr in range(ROW_TILES):
    #     for tc in range(COLUMN_TILES):
    #         # tile = {
    #         #         'gt': tiled_dataset['gt'][tr][tc],
    #         #         'hs': tiled_dataset['hs'][tr][tc],
    #         #         'vhr': tiled_dataset['vhr'][tr][tc],
    #         #         'lidar': tiled_dataset['lidar'][tr][tc],
    #         #     }
    #         tile = tiled_dataset['hs'][tr][tc]
    #         if (tr, tc) in TRAINING_GT_TILES:
    #             training_tiles.append(tile)
    #         else:
    #             testing_tiles.append(tile)



    return full_image_dataset, train_indices, test_indices

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
        x, y = value
        assign_0 = y + pad_length    # Row assignment
        assign_1 = x + pad_length     # Column assignment
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

    full_image_dataset, train_indices, test_indices = load_houston_dataset()

    data = full_image_dataset['hs'][...,::8]
    gt = full_image_dataset['gt']

    TRAIN_VAL_SPLIT = 0.2   # Use 20% of training samples for validation

    ### Set Constants ###
    INPUT_DIMENSION_CONV = data.shape[-1]  # number of spectral bands

    TOTAL_SIZE = len(train_indices) + len(test_indices) # total number of samples across all classes
    VAL_SIZE = int(len(train_indices) * TRAIN_VAL_SPLIT)     # total number of samples in the validation dataset
    TRAIN_SIZE = len(train_indices) - VAL_SIZE          # total number of samples in the training dataset
    TEST_SIZE = len(test_indices) # total number of samples in test set
    VALIDATION_SPLIT = 0.8  # 20% for training and 80% for validation and testing

    # Spatial context size (number of neighbours in each spatial direction)
    PATCH_LENGTH = 1

    ITER = 1        # number of iterations to run this model
    CATEGORY = len(CLASS_LIST)   # number of classification categories in dataset

    ### Set Variables ###

    # Get Indian Pines dataset array
    data_UH = data

    # Get Indian Pines ground truth array
    gt_UH = gt

    new_gt_UH = gt_UH   # copy of ground truth array data
    batch_size = 8      # number of samples to put through model in one shot
    nb_epoch = 15       # number of epochs to run model for

    img_rows = PATCH_LENGTH * 2 + 1 # number of rows in neighborhood
    img_cols = PATCH_LENGTH * 2 + 1 # number of cols in neighborhood
    bands = INPUT_DIMENSION_CONV    # number of spectral bands
    classes = CATEGORY              # number of label categories

    # Number of epochs with no improvement after which training will be
    # stopped
    patience = 200

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
    # train_indices = [(x+PATCH_LENGTH, y+PATCH_LENGTH) for x, y in train_indices]
    # test_indices = [(x+PATCH_LENGTH, y+PATCH_LENGTH) for x, y in test_indices]

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
        overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
        
        # Get prediction confusion matrix
        confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
        
        # Get individual class accuracy as well as average accuracy
        each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
        
        # Get Kappa metric from predictions
        kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])
        
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
