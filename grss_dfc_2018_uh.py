#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Class for loading University of Houston 2018 Data Fusion Contest dataset
for training and visualization.
"""

### Built-in Imports ###
import argparse
import gc
from locale import normalize
import os

### Other Library Imports ###

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import spectral
import wx


### Local Imports ###

### Environment Setup ###

### Global Constants ###

# Path to directory containing all GRSS 2018 Data Fusion Contest
# University of Houston image data
UH_2018_DATASET_DIRECTORY_PATH = 'datasets/grss_dfc_2018/'

# Following paths are assumed to be from the root UH 2018 dataset path
# - DEM_C123_3msr is a bare-earth digital elevation model (DEM) generated
#    from returns classified as ground from all three sensors
# - DEM_C123_TLI is a bare-earth DEM with void filling for manmade structures
# - DEM+B_C123 is a hybrid ground and building DEM, generated from returns
#    that were classified as coming from buildings and the ground by all
#    titan sensors
UH_2018_TRAINING_GT_IMAGE_PATH = 'TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif'
UH_2018_TESTING_GT_IMAGE_PATH = 'TestingGT/Test_Labels.tif'
UH_2018_HS_IMAGE_PATH = 'FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix'
UH_2018_LIDAR_DSM_PATH = 'Lidar GeoTiff Rasters/DSM_C12/UH17c_GEF051.tif'
UH_2018_LIDAR_DEM_3MSR_PATH = 'Lidar GeoTiff Rasters/DEM_C123_3msr/UH17_GEG051.tif'
UH_2018_LIDAR_DEM_TLI_PATH = 'Lidar GeoTiff Rasters/DEM_C123_TLI/UH17_GEG05.tif'
UH_2018_LIDAR_DEM_B_PATH = 'Lidar GeoTiff Rasters/DEM+B_C123/UH17_GEM051.tif'
UH_2018_LIDAR_INTENSITY_1550NM_PATH = 'Lidar GeoTiff Rasters/Intensity_C1/UH17_GI1F051.tif'
UH_2018_LIDAR_INTENSITY_1064NM_PATH = 'Lidar GeoTiff Rasters/Intensity_C2/UH17_GI2F051.tif'
UH_2018_LIDAR_INTENSITY_532NM_PATH = 'Lidar GeoTiff Rasters/Intensity_C3/UH17_GI3F051.tif'

# Paths, in order of tile, for the very high resolution RGB image
UH_2018_VHR_IMAGE_PATHS = [
    [
        'Final RGB HR Imagery/UH_NAD83_271460_3290290.tif',
        'Final RGB HR Imagery/UH_NAD83_272056_3290290.tif',
        'Final RGB HR Imagery/UH_NAD83_272652_3290290.tif',
        'Final RGB HR Imagery/UH_NAD83_273248_3290290.tif',
        'Final RGB HR Imagery/UH_NAD83_273844_3290290.tif',
        'Final RGB HR Imagery/UH_NAD83_274440_3290290.tif',
        'Final RGB HR Imagery/UH_NAD83_275036_3290290.tif',
    ],
    [
        'Final RGB HR Imagery/UH_NAD83_271460_3289689.tif',
        'Final RGB HR Imagery/UH_NAD83_272056_3289689.tif',
        'Final RGB HR Imagery/UH_NAD83_272652_3289689.tif',
        'Final RGB HR Imagery/UH_NAD83_273248_3289689.tif',
        'Final RGB HR Imagery/UH_NAD83_273844_3289689.tif',
        'Final RGB HR Imagery/UH_NAD83_274440_3289689.tif',
        'Final RGB HR Imagery/UH_NAD83_275036_3289689.tif',
    ],
]

# Number of rows and columns in the overall dataset image such that 
# parts of the dataset can be matched with the ground truth
UH_2018_NUM_TILE_COLUMNS = 7
UH_2018_NUM_TILE_ROWS = 2

# List of tuples corresponding to the image tiles that match the ground
# truth for the training set
UH_2018_TRAINING_GT_TILES = ((1,1), (1,2), (1, 3), (1, 4))

# List of tuples corresponding to the image tiles that match the ground
# truth for the testing set
UH_2018_TESTING_GT_TILES = ((0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6),
                            (1,0),                             (1,5), (1,6))

# Mapping of overall dataset training ground truth tile indices to the
# tile indices of the actual ground truth image
UH_2018_TRAINING_GT_TILE_OFFSETS = {
    (1,1) : (0,0),
    (1,2) : (0,1),
    (1,3) : (0,2),
    (1,4) : (0,3)
}

# GSD = Ground Sampling Distance
UH_2018_VHR_GSD = 0.05  # Very high resolution image GSD in meters
UH_2018_HS_GSD = 1.0    # Hyperspectral image GSD in meters
UH_2018_LIDAR_GSD = 0.5 # LiDAR raster image GSD in meters
UH_2018_GT_GSD = 0.5    # Ground truth images GSD in meters

# A list of the wavelength values for each lidar band channel
UH_2018_LIDAR_BAND_WAVELENGTHS = [
    '1550nm',
    '1064nm',
    '532nm'
]

# Number of hyperspectral band channels
UH_2018_NUM_HS_BANDS = 48

# Threshold value for hyperspectral band intensities
UH_2018_HS_BAND_THRES = 100_000

# Number of multispectral LiDAR band channels
UH_2018_NUM_LIDAR_BANDS = 3

# Threshold value for LiDAR multispectral intensities
UH_2018_MS_INTENSITY_THRESHOLD = 1e4

# Threshold value for LiDAR DSM
UH_2018_DSM_THRESHOLD = 1e10

# A list of the wavelength values for each of the hyperspectal band
# channels
UH_2018_HS_BAND_WAVELENGTHS = [
    '374.395nm +/- 7.170nm',
    '388.733nm +/- 7.168nm', 
    '403.068nm +/- 7.167nm', 
    '417.401nm +/- 7.166nm', 
    '431.732nm +/- 7.165nm', 
    '446.061nm +/- 7.164nm', 
    '460.388nm +/- 7.163nm', 
    '474.712nm +/- 7.162nm', 
    '489.035nm +/- 7.161nm', 
    '503.356nm +/- 7.160nm', 
    '517.675nm +/- 7.159nm', 
    '531.992nm +/- 7.158nm', 
    '546.308nm +/- 7.158nm', 
    '560.622nm +/- 7.157nm', 
    '574.936nm +/- 7.156nm', 
    '589.247nm +/- 7.156nm', 
    '603.558nm +/- 7.155nm', 
    '617.868nm +/- 7.155nm', 
    '632.176nm +/- 7.154nm', 
    '646.484nm +/- 7.154nm', 
    '660.791nm +/- 7.153nm', 
    '675.097nm +/- 7.153nm', 
    '689.402nm +/- 7.153nm', 
    '703.707nm +/- 7.152nm', 
    '718.012nm +/- 7.152nm', 
    '732.316nm +/- 7.152nm', 
    '746.620nm +/- 7.152nm', 
    '760.924nm +/- 7.152nm', 
    '775.228nm +/- 7.152nm', 
    '789.532nm +/- 7.152nm', 
    '803.835nm +/- 7.152nm', 
    '818.140nm +/- 7.152nm', 
    '832.444nm +/- 7.152nm', 
    '846.749nm +/- 7.153nm', 
    '861.054nm +/- 7.153nm', 
    '875.360nm +/- 7.153nm', 
    '889.666nm +/- 7.153nm', 
    '903.974nm +/- 7.154nm', 
    '918.282nm +/- 7.154nm', 
    '932.591nm +/- 7.155nm', 
    '946.901nm +/- 7.155nm', 
    '961.212nm +/- 7.156nm', 
    '975.525nm +/- 7.157nm', 
    '989.839nm +/- 7.157nm', 
    '1004.154nm +/- 7.158nm', 
    '1018.471nm +/- 7.159nm', 
    '1032.789nm +/- 7.160nm', 
    '1047.109nm +/- 7.160nm', 
]

# A list of hexidecimal color values corresponding to the wavelength of
# the hyperspectral bands
UH_2018_BAND_RGB = [
    '#610061',  #374nm
    '#780088',  #389nm
    '#8300c0',  #403nm
    '#7100f4',  #417nm
    '#3300ff',  #432nm
    '#002fff',  #446nm
    '#007bff',  #460nm
    '#00c0ff',  #475nm
    '#00fbff',  #489nm
    '#00ff6e',  #503nm
    '#2dff00',  #518nm
    '#65ff00',  #532nm
    '#96ff00',  #546nm
    '#c6ff00',  #561nm
    '#f0ff00',  #575nm
    '#ffe200',  #589nm
    '#ffb000',  #604nm
    '#ff7e00',  #618nm
    '#ff4600',  #632nm
    '#ff0000',  #646nm
    '#fd0000',  #661nm
    '#fb0000',  #675nm
    '#fa0000',  #689nm
    '#f80000',  #704nm
    '#de0000',  #718nm
    '#c40000',  #732nm
    '#a70000',  #747nm
    '#8a0000',  #761nm
    '#6d0000',  #775nm
    '#610000',  #790nm (representation)
    '#5e0000',  #804nm (representation)
    '#5c0000',  #818nm (representation)
    '#590000',  #843nm (representation)
    '#570000',  #847nm (representation)
    '#540000',  #862nm (representation)
    '#510000',  #875nm (representation)
    '#4f0000',  #890nm (representation)
    '#4c0000',  #904nm (representation)
    '#4a0000',  #918nm (representation)
    '#470000',  #933nm (representation)
    '#440000',  #947nm (representation)
    '#420000',  #961nm (representation)
    '#3f0000',  #976nm (representation)
    '#3d0000',  #990nm (representation)
    '#3a0000',  #1004nm (representation)
    '#370000',  #1018nm (representation)
    '#350000',  #1033nm (representation)
    '#320000',  #1047nm (representation)
]

UH_2018_IGNORED_CLASSES = [0]
    
# List of classes where the index is the value of the pixel in the
# ground truth image
UH_2018_CLASS_LIST = [
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

# Map of classes where the key is the value of the pixel in the
# ground truth image
UH_2018_CLASS_MAP = {index: label for index, label in enumerate(UH_2018_CLASS_LIST)}


# Number of class labels for the University of Houston 2018 dataset
# (one is subtracted to exclude the 'undefined' class)
UH_2018_NUM_CLASSES = len(UH_2018_CLASS_LIST)

### Classes ###

class UH_2018_Dataset:
    """
    Class for loading and manipulating different parts of the GRSS 2018
    Data Fusion Contest University of Houston dataset.
    """

    def __init__(self, dataset_path=UH_2018_DATASET_DIRECTORY_PATH):

        # Set dataset attributes
        self.name = 'GRSS_DFC_2018_UH'

        # Set dataset file paths
        self.path_to_dataset_directory = dataset_path
        self.path_to_training_gt_image = UH_2018_TRAINING_GT_IMAGE_PATH
        self.path_to_testing_gt_image = UH_2018_TESTING_GT_IMAGE_PATH
        self.path_to_hs_image = UH_2018_HS_IMAGE_PATH
        self.path_to_lidar_dsm = UH_2018_LIDAR_DSM_PATH
        self.path_to_lidar_dem_3msr = UH_2018_LIDAR_DEM_3MSR_PATH
        self.path_to_lidar_dem_tli = UH_2018_LIDAR_DEM_TLI_PATH
        self.path_to_lidar_dem_b = UH_2018_LIDAR_DEM_B_PATH
        self.path_to_lidar_1550nm_intensity = UH_2018_LIDAR_INTENSITY_1550NM_PATH
        self.path_to_lidar_1064nm_intensity = UH_2018_LIDAR_INTENSITY_1064NM_PATH
        self.path_to_lidar_532nm_intensity = UH_2018_LIDAR_INTENSITY_532NM_PATH

        # Set dataset ground truth attributes
        self.gt_class_label_list = UH_2018_CLASS_LIST
        self.gt_class_value_mapping = UH_2018_CLASS_MAP
        self.gt_num_classes = UH_2018_NUM_CLASSES
        self.gt_ignored_labels = UH_2018_IGNORED_CLASSES

        # Set dataset hyperspectral image attributes
        self.hs_num_bands = UH_2018_NUM_HS_BANDS
        self.hs_band_rgb_list = UH_2018_BAND_RGB
        self.hs_band_wavelength_labels = UH_2018_HS_BAND_WAVELENGTHS
        self.hs_band_val_thres = UH_2018_HS_BAND_THRES

        # Set dataset lidar data attributes
        self.lidar_ms_num_bands = UH_2018_NUM_LIDAR_BANDS
        self.lidar_ms_intensity_thres = UH_2018_MS_INTENSITY_THRESHOLD
        self.lidar_dsm_thres = UH_2018_DSM_THRESHOLD

        # Set dataset VHR RGB image attributes
        #TODO

        # Set miscellaneous dataset attributes
        self.gsd_gt = UH_2018_GT_GSD
        self.gsd_hs = UH_2018_HS_GSD
        self.gsd_lidar = UH_2018_LIDAR_GSD
        self.gsd_vhr = UH_2018_VHR_GSD

        self.dataset_tiled_subset_rows = UH_2018_NUM_TILE_ROWS
        self.dataset_tiled_subset_cols = UH_2018_NUM_TILE_COLUMNS
        self.dataset_training_subset = UH_2018_TRAINING_GT_TILES
        self.dataset_training_subset_map = UH_2018_TRAINING_GT_TILE_OFFSETS
        self.dataset_testing_subset = UH_2018_TESTING_GT_TILES

        # Initialize dataset variables
        self.gt_image = None
        self.gt_image_tiles = None
        self.hs_image = None
        self.hs_image_tiles = None
        self.lidar_ms_image = None
        self.lidar_ms_image_tiles = None
        self.lidar_dsm_image = None
        self.lidar_dsm_image_tiles = None
        self.lidar_dem_image = None
        self.lidar_dem_image_tiles = None
        self.lidar_ndsm_image = None
        self.lidar_ndsm_image_tiles = None
        self.vhr_image = None
        self.vhr_image_tiles = None
    


    def clear_all_images(self):
        """Clears values of all image variables to free memory."""

        # Delete the variables to mark the memory as unused
        del self.gt_image
        del self.gt_image_tiles
        del self.hs_image
        del self.hs_image_tiles
        del self.lidar_ms_image
        del self.lidar_ms_image_tiles
        del self.lidar_dsm_image
        del self.lidar_dsm_image_tiles
        del self.lidar_dem_image
        del self.lidar_dem_image_tiles
        del self.lidar_ndsm_image
        del self.lidar_ndsm_image_tiles
        del self.vhr_image
        del self.vhr_image_tiles

        # Run garbage collection to release the memory
        gc.collect()

        # Reinitialize the variables
        self.gt_image = None
        self.gt_image_tiles = None
        self.hs_image = None
        self.hs_image_tiles = None
        self.lidar_ms_image = None
        self.lidar_ms_image_tiles = None
        self.lidar_dsm_image = None
        self.lidar_dsm_image_tiles = None
        self.lidar_dem_image = None
        self.lidar_dem_image_tiles = None
        self.lidar_ndsm_image = None
        self.lidar_ndsm_image_tiles = None
        self.vhr_image = None
        self.vhr_image_tiles = None



    def clear_gt_images(self):
        """Clears values of ground truth image variables to free memory."""
        # Delete the variables to mark the memory as unused
        del self.gt_image
        del self.gt_image_tiles

        # Run garbage collection to release the memory
        gc.collect()

        # Reinitialize the variables
        self.gt_image = None
        self.gt_image_tiles = None
    


    def clear_hs_images(self):
        """Clears values of hyperspectral image variables to free memory."""

        # Delete the variables to mark the memory as unused
        del self.hs_image
        del self.hs_image_tiles

        # Run garbage collection to release the memory
        gc.collect()

        # Reinitialize the variables
        self.hs_image = None
        self.hs_image_tiles = None
    


    def clear_lidar_ms_images(self):
        """
        Clears values of lidar multispectral image variables to free 
        memory.
        """

        # Delete the variables to mark the memory as unused
        del self.lidar_ms_image
        del self.lidar_ms_image_tiles

        # Run garbage collection to release the memory
        gc.collect()

        # Reinitialize the variables
        self.lidar_ms_image = None
        self.lidar_ms_image_tiles = None
    


    def clear_lidar_dsm_images(self):
        """
        Clears values of lidar digital surface model (NDSM) image
        variables to free memory.
        """

        # Delete the variables to mark the memory as unused
        del self.lidar_dsm_image
        del self.lidar_dsm_image_tiles

        # Run garbage collection to release the memory
        gc.collect()

        # Reinitialize the variables
        self.lidar_dsm_image = None
        self.lidar_dsm_image_tiles = None

    def clear_lidar_dem_images(self):
        """
        Clears values of lidar digital surface model (NDSM) image
        variables to free memory.
        """

        # Delete the variables to mark the memory as unused
        del self.lidar_dem_image
        del self.lidar_dem_image_tiles

        # Run garbage collection to release the memory
        gc.collect()

        # Reinitialize the variables
        self.lidar_dem_image = None
        self.lidar_dem_image_tiles = None

    def clear_lidar_ndsm_images(self):
        """
        Clears values of lidar normalized digital surface model (NDSM)
        image variables to free memory.
        """

        # Delete the variables to mark the memory as unused
        del self.lidar_ndsm_image
        del self.lidar_ndsm_image_tiles

        # Run garbage collection to release the memory
        gc.collect()

        # Reinitialize the variables
        self.lidar_ndsm_image = None
        self.lidar_ndsm_image_tiles = None



    def clear_vhr_images(self):
        """
        Clears values of very high resolution image variables to free 
        memory.
        """

        # Delete the variables to mark the memory as unused
        del self.vhr_image
        del self.vhr_image_tiles

        # Run garbage collection to release the memory
        gc.collect()

        # Reinitialize the variables
        self.vhr_image = None
        self.vhr_image_tiles = None



    def merge_tiles(self, tiles, num_rows = None, num_cols = None):
        """Merges a set of image tiles into a single image."""

        print('Merging image tiles...')

        # If rows or columns are not specified, use defaults
        if num_rows is None: num_rows = self.dataset_tiled_subset_rows
        if num_cols is None: num_cols = self.dataset_tiled_subset_cols

        # Initialize empty list of image row values
        image_rows = []

        # Loop through each tile and stitch them together into single image
        for row in range(0, num_rows):
            # Get first tile in row
            img_row = np.copy(tiles[row * num_cols])

            # Loop through remaining tiles in current row
            for col in range(1, num_cols):
                # Concatenate each subsequent tile in row to image row array
                img_row = np.concatenate((img_row, tiles[row*num_cols + col]), axis=1)
            
            # Append image row to list of image rows
            image_rows.append(img_row)

        # Concatenate all image rows together to create single image
        merged_image = np.concatenate(image_rows, axis=0)

        return merged_image



    def load_full_gt_image(self, train_only=False, test_only=False):
        """
        Loads the full-size ground truth image mask for the University
        of Houston 2018 dataset.
        """

        if train_only:
            print('Loading training ground truth image...')
        elif test_only:
            print('Loading test ground truth image...')
        else:
            print('Loading full ground truth image...')

        # Ground truth can only be loaded as tiles since there's two
        # images, so load tiles and then merge them to create full GT
        # image
        self.gt_image = self.merge_tiles(
            self.load_gt_image_tiles(train_only=train_only, 
                                     test_only=test_only))

        return self.gt_image



    def load_gt_image_tiles(self, tile_list=None, train_only=False, test_only=False):
        """
        Loads the University of Houston 2018 dataset's ground truth
        images as a set of tiles. If no tile list is given, the whole 
        image will be loaded as tiles.
        """

        if train_only:
            print('Loading training ground truth tiles...')
        elif test_only:
            print('Loading test ground truth tiles...')
        else:
            print('Loading training and test ground truth tiles...')

        self.gt_image_tiles = []

        # train_only flag takes priority
        if train_only: test_only = False

        # Get full path to dataset's ground truth images
        train_image_path = os.path.join(self.path_to_dataset_directory,
                                        self.path_to_training_gt_image)
        test_image_path = os.path.join(self.path_to_dataset_directory,
                                       self.path_to_testing_gt_image)

        # Throw error if file path does not exist
        if not os.path.isfile(train_image_path): raise FileNotFoundError(
            f'Path to UH2018 training ground truth image is invalid!'
            f'Path={train_image_path}')
        
        if not os.path.isfile(test_image_path): raise FileNotFoundError(
            f'Path to UH2018 testing ground truth image is invalid!'
            f'Path={test_image_path}')

        with rasterio.open(train_image_path) as train_src, \
             rasterio.open(test_image_path) as test_src:

            # Get the size of the tile windows (use full size test image)
            tile_width = test_src.width / self.dataset_tiled_subset_cols
            tile_height = test_src.height / self.dataset_tiled_subset_rows

                # Read in the image data for each image tile
            for tile_row in range(0, self.dataset_tiled_subset_rows):
                for tile_column in range(0, self.dataset_tiled_subset_cols):

                    # Check to see if current tile is one of the training
                    # ground truth tiles
                    if not test_only and (tile_row, tile_column) in self.dataset_training_subset:

                        offset_row, offset_column = self.dataset_training_subset_map[
                                                        (tile_row, tile_column)]

                        # Set the tile window to read from the image
                        window = Window(tile_width * offset_column ,  
                                        tile_height * offset_row, 
                                        tile_width, tile_height)

                        # Read the tile window from the image
                        tile = train_src.read(1, window = window)

                        # Copy the tile to the tiles array
                        self.gt_image_tiles.append(np.copy(tile))
                    elif not train_only and (tile_row, tile_column) in self.dataset_testing_subset:
                        # Set the tile window to read from the image
                        window = Window(tile_width * tile_column,  
                                        tile_height * tile_row, 
                                        tile_width, tile_height)

                        # Read the tile window from the image
                        tile = test_src.read(1, window = window)

                        # Copy the tile to the tiles array
                        self.gt_image_tiles.append(np.copy(tile))
                    else:
                        self.gt_image_tiles.append(np.zeros(
                            (int(tile_height), int(tile_width)), 
                            dtype=np.uint8))

        return self.gt_image_tiles



    def save_full_gt_image_array(self, path, file_name='full_gt_image.npy'):
        """
        Saves the numpy array of the full ground truth image to a file 
        for faster loading in the future.
        """

        print(f'Saving full ground truth image numpy array to file ({file_name})...')

        # If the gt image member variable is empty, then load the full
        # ground truth image
        if self.gt_image is None: self.load_full_gt_image()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.gt_image)
    


    def save_tiled_gt_image_array(self, path, file_name='tiled_gt_image.npy'):
        """
        Saves the numpy array of the tiled ground truth image to a file 
        for faster loading in the future.
        """

        print(f'Saving full ground truth image numpy array to file ({file_name})...')

        # If the gt image tile member variable is empty, then load all
        # ground truth image tiles
        if self.gt_image_tiles is None: self.load_gt_image_tiles()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.gt_image_tiles)



    def load_full_gt_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset ground truth image.
        """

        print(f'Loading full ground truth image numpy array from file ({file_path})...')

        with open(file_path, 'rb') as infile:
            self.gt_image = np.load(infile)
    


    def load_tiled_gt_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset ground truth image tiles.
        """

        print(f'Loading tiled ground truth image numpy array from file ({file_path})...')

        with open(file_path, 'rb') as infile:
            self.gt_image_tiles = np.load(infile)
    


    def get_gt_class_statistics(self, tiles=None, print_results=False):
        """
        Outputs statistics per each class per ground truth tile (or, if
        no tile or set of tile is specified, the whole ground truth).
        """

        # If the hs image tile member variable is empty, then load all
        # ground truth image tiles
        if self.gt_image_tiles is None: self.load_gt_image_tiles()

        # Initialize statistics dictionary
        statistics = {}

        # If no tiles are specified, use all ground truth tiles
        if tiles is None:
            tiles = []
            for row in range(0, self.dataset_tiled_subset_rows):
                for col in range(0, self.dataset_tiled_subset_cols):
                    tiles.append((row, col))
        
        # Iterate through the tile list to get statistics for each
        # individual tile
        for tile in tiles:
            row, col = tile
            index = row * self.dataset_tiled_subset_cols + col
            
            # Create tile statistics mapping
            tile_statistics = {x:0 for x in range(0,len(self.gt_class_label_list))}

            # Count the class of each pixel in the image tile
            for pixel in np.ravel(self.gt_image_tiles[index]):
                tile_statistics[pixel] += 1
            
            # Create key/value pair for statistics dictionary
            key = f'Tile ({row}, {col})'
            value = [tile_statistics[i] for i in range(1,len(self.gt_class_label_list))]
            
            # Add key value pair to dictionary
            statistics[key] = value

        # Create Pandas DataFrame from statistics dictionary and set the
        # index to be the class labels
        statistics_df = pd.DataFrame(data=statistics)
        statistics_df.index = self.gt_class_label_list[1:]

        # Print out statistics
        if print_results:
            print(statistics_df)
            print()
            print(statistics_df.T.describe())

        return statistics_df



    def load_full_hs_image(self, gsd=UH_2018_GT_GSD,
                           thres=True, normalize=True):
        """
        Loads the full-size hyperspectral image for the University of
        Houston 2018 dataset sampled at the specified GSD.
        """

        print('Loading full hyperspectral image...')

        # Check GSD parameter value
        if gsd <= 0: raise ValueError("'gsd' parameter must be greater than 0!")

        # Set the factor of GSD resampling
        resample_factor = self.gsd_hs / float(gsd)

        # Get full path to dataset's hyperspectral image
        image_path = os.path.join(self.path_to_dataset_directory,
                                  self.path_to_hs_image)

        # Throw error if file path does not exist
        if not os.path.isfile(image_path): raise FileNotFoundError(
            f'Path to UH2018 hyperspectral image is invalid! Path={image_path}')

        # Create image variable
        self.hs_image = None

        # Open the training HSI Envi file as src
        with rasterio.open(image_path, format='ENVI') as src:

            # Set the shape of the resampled image
            out_shape=(src.count,
                        int(src.height * resample_factor), 
                        int(src.width * resample_factor))

            # Read the image, resample it to the appropriate GSD, 
            # arrange the numpy array to be (rows, cols, bands), 
            # and remove unused bands
            self.hs_image = np.moveaxis(src.read(
                        out_shape=out_shape, 
                        resampling=Resampling.nearest), 0, -1)[:,:,:-2]
            
            # Cast image array as float type for normalization
            self.hs_image = self.hs_image.astype(float, copy=False)

            # Threshold the image so that any value over the threshold
            # is set to the image minimum
            if thres:
                self.hs_image[self.hs_image > self.hs_band_val_thres] = self.hs_image.min()

            # Normalize each intensity band between 0.0 and 1.0
            if normalize:
                self.hs_image -= self.hs_image.min()
                self.hs_image /= self.hs_image.max()

        
        return self.hs_image



    def load_hs_image_tiles(self, gsd=UH_2018_GT_GSD, tile_list=None,
                            thres=True, normalize=True):
        """
        Loads the University of Houston 2018 dataset's hyperspectral
        images as a set of tiles sampled at a specified GSD. If no tile
        list is given, the whole image will be loaded as tiles.
        """

        print('Loading hyperspectral image as tiles...')

        # Check GSD parameter value
        if gsd <= 0: raise ValueError("'gsd' parameter must be greater than 0!")
        
        # Check tile_list parameter value
        if tile_list and not isinstance(tile_list, tuple): raise ValueError(
            "'tile_list' parameter should be a tuple of tuples!")

        # Initialize list for tiles
        self.hs_image_tiles = []
        
        # Set the factor of GSD resampling
        resample_factor = self.gsd_hs / float(gsd)

        # Get full path to dataset's hyperspectral image
        image_path = os.path.join(self.path_to_dataset_directory,
                                  self.path_to_hs_image)

        # Throw error if file path does not exist
        if not os.path.isfile(image_path): raise FileNotFoundError(
            f'Path to UH2018 hyperspectral image is invalid! Path={image_path}')

        # Open the training HSI Envi file as src
        with rasterio.open(image_path, format='ENVI') as src:
            # Get the size of the tile windows
            tile_width = src.width / self.dataset_tiled_subset_cols
            tile_height = src.height / self.dataset_tiled_subset_rows

            # Read in the image data for each image tile
            for tile_row in range(0, self.dataset_tiled_subset_rows):
                for tile_column in range(0, self.dataset_tiled_subset_cols):

                    # If specified tiles are desired, then skip any
                    # tiles that do not match the tile_list parameter
                    if tile_list and (tile_row, tile_column) not in tile_list:
                        continue

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
                    self.hs_image_tiles.append(np.copy(tile))

        # If no tiles were added to the tile list, then set image tiles
        # variable to 'None'
        if len(self.hs_image_tiles) == 0: self.hs_image_tiles = None
        else:
            # Turn list of numpy arrays into single numpy array
            self.hs_image_tiles = np.stack(self.hs_image_tiles)

            # Cast image array as float type for normalization
            self.hs_image_tiles = self.hs_image_tiles.astype(float, copy=False)

            # Threshold the image tiles so that any value over the threshold
            # is set to the image minimum
            if thres:
                self.hs_image_tiles[self.hs_image_tiles > self.hs_band_val_thres] = self.hs_image_tiles.min()

            # Normalize each intensity band between 0.0 and 1.0
            if normalize:
                self.hs_image_tiles -= self.hs_image_tiles.min()
                self.hs_image_tiles /= self.hs_image_tiles.max()

        return self.hs_image_tiles



    def save_full_hs_image_array(self, path, file_name='full_hs_image.npy'):
        """
        Saves the numpy array of the full hyperspectral image to a file 
        for faster loading in the future.
        """

        # If the hs image member variable is empty, then load the full
        # hyperspectral image
        if self.hs_image is None: self.load_full_hs_image()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.hs_image)
    


    def save_tiled_hs_image_array(self, path, file_name='tiled_hs_image.npy'):
        """
        Saves the numpy array of the tiled hyperspectral image to a file 
        for faster loading in the future.
        """

        # If the hs image tile member variable is empty, then load all
        # hyperspectral image tiles
        if self.hs_image_tiles is None: self.load_hs_image_tiles()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.hs_image_tiles)




    def load_full_hs_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset hyperspectral image.
        """

        with open(file_path, 'rb') as infile:
            self.hs_image = np.load(infile)
    


    def load_tiled_hs_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset hyperspectral image tiles.
        """

        with open(file_path, 'rb') as infile:
            self.hs_image_tiles = np.load(infile)



    def show_hs_image(self, rgb_channels=None, size=(15,9), 
                      full_gt_overlay=False, 
                      train_gt_overlay=False,
                      test_gt_overlay=False):
        """
        Displays the hyperspectral image using the specified band
        channels as the rgb values.
        """

        # If the hs image member variable is empty, then load the full
        # hyperspectral image
        if self.hs_image is None: self.load_full_hs_image()
        
        image = self.hs_image[:,:,:]

        if full_gt_overlay:
            # If the hs image member variable is empty, then load the full
            # ground truth image
            if self.gt_image is None: self.load_full_gt_image()

            classes = self.gt_image
            title = 'Hyperspectral image w/ ground truth overlay'
        elif test_gt_overlay or train_gt_overlay:
            # If the hs image member variable is empty, then load the
            # ground truth image tiles
            if self.gt_image_tiles is None: self.load_gt_image_tiles()

            # create a copy of ground truth image tiles
            gt_tiles = self.gt_image_tiles.copy()

            # Get tile dimensions
            tile_shape = gt_tiles[0].shape

            # Choose which set of tiles to set to zero values and 
            # set proper image title
            if train_gt_overlay: 
                tiles_to_remove = self.dataset_testing_subset
                title = 'Hyperspectral image w/ training ground truth overlay'
            else: 
                tiles_to_remove = self.dataset_training_subset
                title = 'Hyperspectral image w/ testing ground truth overlay'

            # Zero out tiles not in the desired subset
            for tile in tiles_to_remove:
                row, col = tile
                index = row * self.dataset_tiled_subset_cols + col
                gt_tiles[index] = np.zeros(tile_shape, dtype=np.uint8)
            
            # Create single ground truth image mask
            classes = self.merge_tiles(gt_tiles)

        else:
            classes = None
            title = 'Hyperspectral image'

        plt.close('all')

        view = spectral.imshow(image, 
                               source=image,
                               bands=rgb_channels,
                               classes=classes,
                               figsize=size)
        if (full_gt_overlay 
            or test_gt_overlay 
            or train_gt_overlay): view.set_display_mode('overlay')
        
        view.set_title(title)

        plt.show(block=True)



    def visualize_hs_data_cube(self, size=(1200,900)):
        """
        Creates 3-D visualization of the hyperspectral data cube for
        the hyperspectral image.
        """

        # If the hs image member variable is empty, then load the full
        # hyperspectral image
        if self.hs_image is None: self.load_full_hs_image()

        # Setup WxApp to display 3D spectral cube
        app = wx.App(False)

        # View 3D hyperspectral cube image
        spectral.view_cube(self.hs_image, size=size)

        # Prevent app from closing immediately
        app.MainLoop()



    def load_full_lidar_ms_image(self, gsd=UH_2018_GT_GSD,
                                 thres=True, normalize=True):
        """
        Loads the full-size lidar multispectral intensisty image for the 
        University of Houston 2018 dataset sampled at the specified GSD.
        """

        print('Loading full LiDAR multispectral intensity image...')

        # Check GSD parameter value
        if gsd <= 0: raise ValueError("'gsd' parameter must be greater than 0!")

        # Set the factor of GSD resampling
        resample_factor = self.gsd_lidar / float(gsd)

        # Get full path to dataset's hyperspectral image
        c1_image_path = os.path.join(self.path_to_dataset_directory,
                                     self.path_to_lidar_1550nm_intensity)
        c2_image_path = os.path.join(self.path_to_dataset_directory,
                                     self.path_to_lidar_1064nm_intensity)
        c3_image_path = os.path.join(self.path_to_dataset_directory,
                                     self.path_to_lidar_532nm_intensity)
        

        # Throw error if file paths do not exist
        if not os.path.isfile(c1_image_path): raise FileNotFoundError(
            f'Path to UH2018 1550nm LiDAR intensity image is invalid! Path={c1_image_path}')
        if not os.path.isfile(c2_image_path): raise FileNotFoundError(
            f'Path to UH2018 1064nm LiDAR intensity image is invalid! Path={c2_image_path}')
        if not os.path.isfile(c3_image_path): raise FileNotFoundError(
            f'Path to UH2018 532nm LiDAR intensity image is invalid! Path={c3_image_path}')
        

        # Create image variable
        self.lidar_ms_image = None

        # Open the LiDAR multispectral intensity image files as c1_src
        # (1550nm), c2_src (1064nm), and c3_src (532nm)
        with rasterio.open(c1_image_path) as c1_src, \
             rasterio.open(c2_image_path) as c2_src, \
             rasterio.open(c3_image_path) as c3_src:

            # Set the shape of the resampled image
            c1_out_shape=(c1_src.count,
                        int(c1_src.height * resample_factor), 
                        int(c1_src.width * resample_factor))
            c2_out_shape=(c2_src.count,
                        int(c2_src.height * resample_factor), 
                        int(c2_src.width * resample_factor))
            c3_out_shape=(c3_src.count,
                        int(c3_src.height * resample_factor), 
                        int(c3_src.width * resample_factor))

            # Read the image, resample it to the appropriate GSD, 
            # arrange the numpy array to be (rows, cols, bands)
            c1 = np.moveaxis(c1_src.read(out_shape=c1_out_shape, 
                              resampling=Resampling.nearest), 0, -1)
            c2 = np.moveaxis(c2_src.read(out_shape=c2_out_shape, 
                              resampling=Resampling.nearest), 0, -1)
            c3 = np.moveaxis(c3_src.read(out_shape=c3_out_shape, 
                              resampling=Resampling.nearest), 0, -1)
            
            # Stack each intensity band into a single cube
            self.lidar_ms_image = np.dstack((c1, c2, c3))

            # Threshold the image so that any value over the threshold
            # is set to the image minimum
            if thres:
                self.lidar_ms_image[self.lidar_ms_image > self.lidar_ms_intensity_thres] = self.lidar_ms_image.min()

            # Normalize each intensity band between 0.0 and 1.0
            if normalize:
                self.lidar_ms_image -= self.lidar_ms_image.min()
                self.lidar_ms_image /= self.lidar_ms_image.max()

        return self.lidar_ms_image



    def load_lidar_ms_image_tiles(self, gsd=UH_2018_GT_GSD, tile_list=None,
                                  thres=True, normalize=True):
        """
        Loads the University of Houston 2018 dataset's lidar
        multispectral intensity image as a set of tiles sampled at a 
        specified GSD. If no tile list is given, the whole image will be 
        loaded as tiles.
        """

        print('Loading LiDAR multispectral intensity image tiles...')

        # Check GSD parameter value
        if gsd <= 0: raise ValueError("'gsd' parameter must be greater than 0!")

        # Check tile_list parameter value
        if tile_list and not isinstance(tile_list, tuple): raise ValueError(
            "'tile_list' parameter should be a tuple of tuples!")

        # Set the factor of GSD resampling
        resample_factor = self.gsd_lidar / float(gsd)

        # Get full path to dataset's hyperspectral image
        c1_image_path = os.path.join(self.path_to_dataset_directory,
                                     self.path_to_lidar_1550nm_intensity)
        c2_image_path = os.path.join(self.path_to_dataset_directory,
                                     self.path_to_lidar_1064nm_intensity)
        c3_image_path = os.path.join(self.path_to_dataset_directory,
                                     self.path_to_lidar_532nm_intensity)
        

        # Throw error if file paths do not exist
        if not os.path.isfile(c1_image_path): raise FileNotFoundError(
            f'Path to UH2018 1550nm LiDAR intensity image is invalid! Path={c1_image_path}')
        if not os.path.isfile(c2_image_path): raise FileNotFoundError(
            f'Path to UH2018 1064nm LiDAR intensity image is invalid! Path={c2_image_path}')
        if not os.path.isfile(c3_image_path): raise FileNotFoundError(
            f'Path to UH2018 532nm LiDAR intensity image is invalid! Path={c3_image_path}')
        

        # Create image variable
        self.lidar_ms_image_tiles = []

        # Open the LiDAR multispectral intensity image files as c1_src
        # (1550nm), c2_src (1064nm), and c3_src (532nm)
        with rasterio.open(c1_image_path) as c1_src, \
             rasterio.open(c2_image_path) as c2_src, \
             rasterio.open(c3_image_path) as c3_src:

            # Get the size of the tile windows
            c1_tile_width = c1_src.width / self.dataset_tiled_subset_cols
            c1_tile_height = c1_src.height / self.dataset_tiled_subset_rows
            c2_tile_width = c1_src.width / self.dataset_tiled_subset_cols
            c2_tile_height = c1_src.height / self.dataset_tiled_subset_rows
            c3_tile_width = c1_src.width / self.dataset_tiled_subset_cols
            c3_tile_height = c1_src.height / self.dataset_tiled_subset_rows

            # Read in the image data for each image tile
            for tile_row in range(0, self.dataset_tiled_subset_rows):
                for tile_column in range(0, self.dataset_tiled_subset_cols):

                    # If specified tiles are desired, then skip any
                    # tiles that do not match the tile_list parameter
                    if tile_list and (tile_row, tile_column) not in tile_list:
                        continue

                    # Set the tile window to read from the image
                    c1_window = Window(c1_tile_width * tile_column,  
                                    c1_tile_height * tile_row, 
                                    c1_tile_width, c1_tile_height)
                    c2_window = Window(c2_tile_width * tile_column,  
                                    c2_tile_height * tile_row, 
                                    c2_tile_width, c2_tile_height)
                    c3_window = Window(c3_tile_width * tile_column,  
                                    c3_tile_height * tile_row, 
                                    c3_tile_width, c3_tile_height)

                    # Set the shape of the resampled tile
                    c1_out_shape=(c1_src.count,
                        int(c1_src.height * resample_factor), 
                        int(c1_src.width * resample_factor))
                    c2_out_shape=(c2_src.count,
                                int(c2_src.height * resample_factor), 
                                int(c2_src.width * resample_factor))
                    c3_out_shape=(c3_src.count,
                                int(c3_src.height * resample_factor), 
                                int(c3_src.width * resample_factor))

                    # Read the tile window from the image, resample it to
                    # the appropriate GSD, arrange the numpy array to be
                    # (rows, cols, bands)
                    c1_tile = np.moveaxis(c1_src.read(
                                    window = c1_window, 
                                    out_shape=c1_out_shape, 
                                    resampling=Resampling.nearest), 0, -1)
                    c2_tile = np.moveaxis(c2_src.read(
                                    window = c2_window, 
                                    out_shape=c2_out_shape, 
                                    resampling=Resampling.nearest), 0, -1)
                    c3_tile = np.moveaxis(c3_src.read(
                                    window = c3_window, 
                                    out_shape=c3_out_shape, 
                                    resampling=Resampling.nearest), 0, -1)
                    
                    # Copy the tile to the tiles array
                    self.lidar_ms_image_tiles.append(np.dstack((c1_tile, 
                                                                c2_tile, 
                                                                c3_tile)))

        # If no tiles were added to the tile list, then set image tiles
        # variable to 'None'
        if len(self.lidar_ms_image_tiles) == 0: self.lidar_ms_image_tiles = None
        else:
            # Turn list of numpy arrays into single numpy array
            self.lidar_ms_image_tiles = np.stack(self.lidar_ms_image_tiles)

            # Threshold the image tiles so that any value over the threshold
            # is set to the image minimum
            if thres:
                self.lidar_ms_image_tiles[self.lidar_ms_image_tiles > self.lidar_ms_intensity_thres] = self.lidar_ms_image_tiles.min()

            # Normalize each intensity band between 0.0 and 1.0
            if normalize:
                self.lidar_ms_image_tiles -= self.lidar_ms_image_tiles.min()
                self.lidar_ms_image_tiles /= self.lidar_ms_image_tiles.max()

        return self.lidar_ms_image_tiles



    def save_full_lidar_ms_image_array(self, path, file_name='full_lidar_multispectral_image.npy'):
        """
        Saves the numpy array of the full lidar multispectral image to a 
        file for faster loading in the future.
        """

        # If the lidar multispectral intensity image member variable is 
        # empty, then load the full lidar multispectral intensity image
        if self.lidar_ms_image is None: self.load_full_lidar_ms_image()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.lidar_ms_image)
    


    def save_tiled_lidar_ms_image_array(self, path, file_name='tiled_lidar_multispectral_image.npy'):
        """
        Saves the numpy array of the tiled lidar multispectral image to 
        a file for faster loading in the future.
        """

        # If the lidar multispectral intensity image tile member
        # variable is empty, then load all lidar multispectral intensity
        # image tiles
        if self.lidar_ms_image_tiles is None: self.load_lidar_ms_image_tiles()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.lidar_ms_image_tiles)




    def load_full_lidar_ms_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset lidar multispectral intensity image.
        """

        with open(file_path, 'rb') as infile:
            self.lidar_ms_image = np.load(infile)
    


    def load_tiled_lidar_ms_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset lidar multispectral intensity image tiles.
        """

        with open(file_path, 'rb') as infile:
            self.lidar_ms_image_tiles = np.load(infile)



    def show_lidar_ms_image(self, size=(15,9), 
                            full_gt_overlay=False, 
                            train_gt_overlay=False,
                            test_gt_overlay=False):
        """
        Displays the multispectral lidar image in RGB with red=1550nm,
        green=1064nm, and blue=532nm bands, with optional ground truth 
        overlay on the image.
        """

        # If the LiDAR multispectral intensity image member variable is
        # empty, then load the full lidar multispectral intensity image
        if self.lidar_ms_image is None: self.load_full_lidar_ms_image()

        image = self.lidar_ms_image[:,:,:]

        if full_gt_overlay:
            # If the ground truth image member variable is empty, then 
            # load the full ground truth image
            if self.gt_image is None: self.load_full_gt_image()

            classes = self.gt_image
            title = 'LiDAR multispectral intensity image w/ ground truth overlay'
        elif test_gt_overlay or train_gt_overlay:
            # If the ground truth image tiles member variable is empty, 
            # then load the ground truth image tiles
            if self.gt_image_tiles is None: self.load_gt_image_tiles()

            # create a copy of ground truth image tiles
            gt_tiles = self.gt_image_tiles.copy()

            # Get tile dimensions
            tile_shape = gt_tiles[0].shape

            # Choose which set of tiles to set to zero values and 
            # set proper image title
            if train_gt_overlay: 
                tiles_to_remove = self.dataset_testing_subset
                title = 'LiDAR multispectral intensity image w/ training ground truth overlay'
            else: 
                tiles_to_remove = self.dataset_training_subset
                title = 'LiDAR multispectral intensity image w/ testing ground truth overlay'

            # Zero out tiles not in the desired subset
            for tile in tiles_to_remove:
                row, col = tile
                index = row * self.dataset_tiled_subset_cols + col
                gt_tiles[index] = np.zeros(tile_shape, dtype=np.uint8)
            
            # Create single ground truth image mask
            classes = self.merge_tiles(gt_tiles)

        else:
            classes = None
            title = 'LiDAR multispectral intensity image'

        plt.close('all')

        view = spectral.imshow(image, 
                               source=image,
                               classes=classes,
                               figsize=size)
        if (full_gt_overlay 
            or test_gt_overlay 
            or train_gt_overlay): view.set_display_mode('overlay')
        
        view.set_title(title)

        plt.show(block=True)
    


    def load_full_lidar_dsm_image(self, gsd=UH_2018_GT_GSD,
                                  thres=True, normalize=True):
        """
        Loads the full-size LiDAR digital surface model (DSM) image for 
        the University of Houston 2018 dataset sampled at the specified 
        GSD.
        """

        print('Loading full LiDAR DSM image...')

        # Check GSD parameter value
        if gsd <= 0: raise ValueError("'gsd' parameter must be greater than 0!")

        # Set the factor of GSD resampling
        resample_factor = self.gsd_lidar / float(gsd)

        # Get full path to dataset's LiDAR DSM image
        image_path = os.path.join(self.path_to_dataset_directory,
                                  self.path_to_lidar_dsm)

        # Throw error if file paths do not exist
        if not os.path.isfile(image_path): raise FileNotFoundError(
            f'Path to UH2018 LiDAR DSM image is invalid! Path={image_path}')
        
        # Create image variable
        self.lidar_dsm_image = None

        # Open the LiDAR DSM file as src
        with rasterio.open(image_path) as src:

            # Set the shape of the resampled image
            out_shape=(src.count,
                       int(src.height * resample_factor), 
                       int(src.width * resample_factor))

            # Read the image, resample it to the appropriate GSD, 
            # arrange the numpy array to be (rows, cols, bands), 
            # and remove unused bands
            # Read the image, resample it to the appropriate GSD, 
            # arrange the numpy array to be (rows, cols, bands), 
            # and remove unused bands
            self.lidar_dsm_image = np.moveaxis(src.read(
                                               out_shape=out_shape, 
                                               resampling=Resampling.nearest), 0, -1)

            # Threshold the image so that any value over the threshold
            # is set to the image minimum
            if thres:
                self.lidar_dsm_image[self.lidar_dsm_image > self.lidar_dsm_thres] = self.lidar_dsm_image.min()

            # Normalize each intensity band between 0.0 and 1.0
            if normalize:
                self.lidar_dsm_image -= self.lidar_dsm_image.min()
                self.lidar_dsm_image /= self.lidar_dsm_image.max()

        return self.lidar_dsm_image



    def load_lidar_dsm_image_tiles(self, gsd=UH_2018_GT_GSD, tile_list=None,
                                   thres=True, normalize=True):
        """
        Loads the University of Houston 2018 dataset's LiDAR digital
        surface model (DSM) image as a set of tiles sampled at a 
        specified GSD. If no tile list is given, the whole image will be 
        loaded as tiles.
        """

        print('Loading LiDAR DSM image tiles...')

        # Check GSD parameter value
        if gsd <= 0: raise ValueError("'gsd' parameter must be greater than 0!")

        # Check tile_list parameter value
        if tile_list and not isinstance(tile_list, tuple): raise ValueError(
            "'tile_list' parameter should be a tuple of tuples!")

        # Set the factor of GSD resampling
        resample_factor = self.gsd_lidar / float(gsd)

        # Get full path to dataset's LiDAR DSM image
        image_path = os.path.join(self.path_to_dataset_directory,
                                  self.path_to_lidar_dsm)

        # Throw error if file paths do not exist
        if not os.path.isfile(image_path): raise FileNotFoundError(
            f'Path to UH2018 LiDAR DSM image is invalid! Path={image_path}')
        
        # Create image variable
        self.lidar_dsm_image_tiles = []

        # Open the LiDAR DSM file as src
        with rasterio.open(image_path) as src:

            # Set the shape of the resampled image
            out_shape=(src.count,
                       int(src.height * resample_factor), 
                       int(src.width * resample_factor))

            # Get the size of the tile windows
            tile_width = src.width / self.dataset_tiled_subset_cols
            tile_height = src.height / self.dataset_tiled_subset_rows

            # Read in the image data for each image tile
            for tile_row in range(0, self.dataset_tiled_subset_rows):
                for tile_column in range(0, self.dataset_tiled_subset_cols):

                    # If specified tiles are desired, then skip any
                    # tiles that do not match the tile_list parameter
                    if tile_list and (tile_row, tile_column) not in tile_list:
                        continue

                    # Set the tile window to read from the image
                    window = Window(tile_width * tile_column,  
                                    tile_height * tile_row, 
                                    tile_width, tile_height)

                    # Set the shape of the resampled tile
                    out_shape=(src.count,
                               int(src.height * resample_factor), 
                               int(src.width * resample_factor))

                    # Read the tile window from the image, resample it to
                    # the appropriate GSD, arrange the numpy array to be
                    # (rows, cols, bands), and remove unused bands
                    tile = np.moveaxis(src.read(
                                       window = window, 
                                       out_shape=out_shape, 
                                       resampling=Resampling.nearest), 0, -1)
                    
                    # Copy the tile to the tiles array
                    self.lidar_dsm_image_tiles.append(tile)

        # If no tiles were added to the tile list, then set image tiles
        # variable to 'None'
        if len(self.lidar_dsm_image_tiles) == 0: self.lidar_dsm_image_tiles = None
        else:
            # Turn list of numpy arrays into single numpy array
            self.lidar_dsm_image_tiles = np.stack(self.lidar_dsm_image_tiles)

            # Threshold the image tiles so that any value over the threshold
            # is set to the image minimum
            if thres:
                self.lidar_dsm_image_tiles[self.lidar_dsm_image_tiles > self.lidar_dsm_thres] = self.lidar_dsm_image_tiles.min()

            # Normalize each intensity band between 0.0 and 1.0
            if normalize:
                self.lidar_dsm_image_tiles -= self.lidar_dsm_image_tiles.min()
                self.lidar_dsm_image_tiles /= self.lidar_dsm_image_tiles.max()

        return self.lidar_dsm_image_tiles



    def save_full_lidar_dsm_image_array(self, path, file_name='full_lidar_dsm_image.npy'):
        """
        Saves the numpy array of the full LiDAR dsm image to a 
        file for faster loading in the future.
        """

        # If the lidar dsm image member variable is empty, then load the full
        # lidar image
        if self.lidar_dsm_image is None: self.load_full_lidar_dsm_image()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.lidar_dsm_image)
    


    def save_tiled_lidar_dsm_image_array(self, path, file_name='tiled_lidar_dsm_image.npy'):
        """
        Saves the numpy array of the tiled LiDAR dsm image to 
        a file for faster loading in the future.
        """

        # If the lidar dsm image tile member variable is empty, then load all
        # lidar dsm image tiles
        if self.lidar_dsm_image_tiles is None: self.load_lidar_dsm_image_tiles()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.lidar_dsm_image_tiles)




    def load_full_lidar_dsm_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset LiDAR digital surface model image.
        """

        with open(file_path, 'rb') as infile:
            self.lidar_dsm_image = np.load(infile)
    


    def load_tiled_lidar_dsm_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset LiDAR digital surface model image tiles.
        """

        with open(file_path, 'rb') as infile:
            self.lidar_dsm_image_tiles = np.load(infile)



    def show_lidar_dsm_image(self, size=(15,9), 
                            full_gt_overlay=False, 
                            train_gt_overlay=False,
                            test_gt_overlay=False):
        """
        Displays the LiDAR dsm image with optional ground truth 
        overlay on the image.
        """

        # If the lidar dsm image member variable is empty, then load the
        # full lidar dsm image
        if self.lidar_dsm_image is None: self.load_full_lidar_dsm_image()

        image = self.lidar_dsm_image[:,:,:]

        if full_gt_overlay:
            # If the ground truth image member variable is empty, then 
            # load the full ground truth image
            if self.gt_image is None: self.load_full_gt_image()

            classes = self.gt_image
            title = 'LiDAR Digital Surface Model (DSM) image w/ ground truth overlay'
        elif test_gt_overlay or train_gt_overlay:
            # If the ground truth tiles image member variable is empty, 
            # then load the ground truth image tiles
            if self.gt_image_tiles is None: self.load_gt_image_tiles()

            # create a copy of ground truth image tiles
            gt_tiles = self.gt_image_tiles.copy()

            # Get tile dimensions
            tile_shape = gt_tiles[0].shape

            # Choose which set of tiles to set to zero values and 
            # set proper image title
            if train_gt_overlay: 
                tiles_to_remove = self.dataset_testing_subset
                title = 'LiDAR Digital Surface Model (DSM) image w/ training ground truth overlay'
            else: 
                tiles_to_remove = self.dataset_training_subset
                title = 'LiDAR Digital Surface Model (DSM) image w/ testing ground truth overlay'

            # Zero out tiles not in the desired subset
            for tile in tiles_to_remove:
                row, col = tile
                index = row * self.dataset_tiled_subset_cols + col
                gt_tiles[index] = np.zeros(tile_shape, dtype=np.uint8)
            
            # Create single ground truth image mask
            classes = self.merge_tiles(gt_tiles)

        else:
            classes = None
            title = 'LiDAR Digital Surface Model (DSM) image'

        plt.close('all')

        view = spectral.imshow(image, 
                               source=image,
                               classes=classes,
                               figsize=size)
        if (full_gt_overlay 
            or test_gt_overlay 
            or train_gt_overlay): view.set_display_mode('overlay')
        
        view.set_title(title)

        plt.show(block=True)



    def load_full_lidar_dem_image(self, gsd=UH_2018_GT_GSD,
                                  use_void_filling_model = False,
                                  use_hybrid_model = True,
                                  thres=True, normalize=True):
        """
        Loads the full-size lidar digital elevation model (DEM) image 
        for the University of Houston 2018 dataset sampled at the 
        specified GSD.
        """

        if use_hybrid_model: 
            dem_path = self.path_to_lidar_dem_b
            print('Loading full LiDAR hybrid DEM image...')
        elif use_void_filling_model: 
            dem_path = self.path_to_lidar_dem_tli
            print('Loading full LiDAR bare-earth elevation w/ void filling DEM image...')
        else:
            dem_path = self.path_to_lidar_dem_3msr
            print('Loading full LiDAR bare-earth elevation DEM image...')

        # Check GSD parameter value
        if gsd <= 0: raise ValueError("'gsd' parameter must be greater than 0!")

        # Set the factor of GSD resampling
        resample_factor = self.gsd_lidar / float(gsd)

        # Get full path to dataset's LiDAR DEM image
        image_path = os.path.join(self.path_to_dataset_directory, dem_path)

        # Throw error if file paths do not exist
        if not os.path.isfile(image_path): raise FileNotFoundError(
            f'Path to UH2018 LiDAR DEM image is invalid! Path={image_path}')
        
        # Create image variable
        self.lidar_dem_image = None

        # Open the LiDAR DSM file as src
        with rasterio.open(image_path) as src:

            # Set the shape of the resampled image
            out_shape=(src.count,
                       int(src.height * resample_factor), 
                       int(src.width * resample_factor))

            # Read the image, resample it to the appropriate GSD, 
            # arrange the numpy array to be (rows, cols, bands), 
            # and remove unused bands
            # Read the image, resample it to the appropriate GSD, 
            # arrange the numpy array to be (rows, cols, bands), 
            # and remove unused bands
            self.lidar_dem_image = np.moveaxis(src.read(
                                               out_shape=out_shape, 
                                               resampling=Resampling.nearest), 0, -1)

            # Threshold the image so that any value over the threshold
            # is set to the image minimum
            if thres:
                self.lidar_dem_image[self.lidar_dem_image > self.lidar_dsm_thres] = self.lidar_dem_image.min()

            # Normalize each intensity band between 0.0 and 1.0
            if normalize:
                self.lidar_dem_image -= self.lidar_dem_image.min()
                self.lidar_dem_image /= self.lidar_dem_image.max()

        return self.lidar_dem_image



    def load_lidar_dem_image_tiles(self, gsd=UH_2018_GT_GSD, tile_list=None,
                                   use_void_filling_model = False,
                                   use_hybrid_model = True,
                                   thres=True, normalize=True):
        """
        Loads the University of Houston 2018 dataset's lidar digital
        elevation model (DEM) image as a set of tiles sampled at a 
        specified GSD. If no tile list is given, the whole image will be 
        loaded as tiles.
        """

        if use_hybrid_model: 
            dem_path = self.path_to_lidar_dem_b
            print('Loading full LiDAR hybrid DEM image...')
        elif use_void_filling_model: 
            dem_path = self.path_to_lidar_dem_tli
            print('Loading full LiDAR bare-earth elevation w/ void filling DEM image...')
        else:
            dem_path = self.path_to_lidar_dem_3msr
            print('Loading full LiDAR bare-earth elevation DEM image...')

        # Check GSD parameter value
        if gsd <= 0: raise ValueError("'gsd' parameter must be greater than 0!")

        # Check tile_list parameter value
        if tile_list and not isinstance(tile_list, tuple): raise ValueError(
            "'tile_list' parameter should be a tuple of tuples!")

        # Set the factor of GSD resampling
        resample_factor = self.gsd_lidar / float(gsd)

        # Get full path to dataset's LiDAR DSM image
        image_path = os.path.join(self.path_to_dataset_directory, dem_path)

        # Throw error if file paths do not exist
        if not os.path.isfile(image_path): raise FileNotFoundError(
            f'Path to UH2018 LiDAR DEM image is invalid! Path={image_path}')
        
        # Create image variable
        self.lidar_dem_image_tiles = []

        # Open the LiDAR DSM file as src
        with rasterio.open(image_path) as src:

            # Set the shape of the resampled image
            out_shape=(src.count,
                       int(src.height * resample_factor), 
                       int(src.width * resample_factor))

            # Get the size of the tile windows
            tile_width = src.width / self.dataset_tiled_subset_cols
            tile_height = src.height / self.dataset_tiled_subset_rows

            # Read in the image data for each image tile
            for tile_row in range(0, self.dataset_tiled_subset_rows):
                for tile_column in range(0, self.dataset_tiled_subset_cols):

                    # If specified tiles are desired, then skip any
                    # tiles that do not match the tile_list parameter
                    if tile_list and (tile_row, tile_column) not in tile_list:
                        continue

                    # Set the tile window to read from the image
                    window = Window(tile_width * tile_column,  
                                    tile_height * tile_row, 
                                    tile_width, tile_height)

                    # Set the shape of the resampled tile
                    out_shape=(src.count,
                               int(src.height * resample_factor), 
                               int(src.width * resample_factor))

                    # Read the tile window from the image, resample it to
                    # the appropriate GSD, arrange the numpy array to be
                    # (rows, cols, bands), and remove unused bands
                    tile = np.moveaxis(src.read(
                                       window = window, 
                                       out_shape=out_shape, 
                                       resampling=Resampling.nearest), 0, -1)
                    
                    # Copy the tile to the tiles array
                    self.lidar_dem_image_tiles.append(tile)

        # If no tiles were added to the tile list, then set image tiles
        # variable to 'None'
        if len(self.lidar_dem_image_tiles) == 0: self.lidar_dem_image_tiles = None
        else:
            # Turn list of numpy arrays into single numpy array
            self.lidar_dem_image_tiles = np.stack(self.lidar_dem_image_tiles)

            # Threshold the image tiles so that any value over the threshold
            # is set to the image minimum
            if thres:
                self.lidar_dem_image_tiles[self.lidar_dem_image_tiles > self.lidar_dsm_thres] = self.lidar_dem_image_tiles.min()

            # Normalize each intensity band between 0.0 and 1.0
            if normalize:
                self.lidar_dem_image_tiles -= self.lidar_dem_image_tiles.min()
                self.lidar_dem_image_tiles /= self.lidar_dem_image_tiles.max()

        return self.lidar_dem_image_tiles



    def save_full_lidar_dem_image_array(self, path, file_name='full_lidar_dem_image.npy'):
        """
        Saves the numpy array of the full lidar dem image to a 
        file for faster loading in the future.
        """

        # If the lidar dsm image member variable is empty, then load the full
        # lidar image
        if self.lidar_dem_image is None: self.load_full_lidar_dem_image()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.lidar_dsm_image)
    


    def save_tiled_lidar_dem_image_array(self, path, file_name='tiled_lidar_dem_image.npy'):
        """
        Saves the numpy array of the tiled lidar dem image to 
        a file for faster loading in the future.
        """

        # If the lidar dsm image tile member variable is empty, then load all
        # lidar dsm image tiles
        if self.lidar_dem_image_tiles is None: self.load_lidar_dem_image_tiles()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.lidar_dsm_image_tiles)




    def load_full_lidar_dem_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset lidar digital elevation model image.
        """

        with open(file_path, 'rb') as infile:
            self.lidar_dem_image = np.load(infile)
    


    def load_tiled_lidar_dem_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset lidar digital elevation model image tiles.
        """

        with open(file_path, 'rb') as infile:
            self.lidar_dsm_image_tiles = np.load(infile)



    def show_lidar_dem_image(self, size=(15,9), 
                            full_gt_overlay=False, 
                            train_gt_overlay=False,
                            test_gt_overlay=False):
        """
        Displays the lidar dem image with optional ground truth 
        overlay on the image.
        """

        # If the lidar dem image member variable is empty, then load the
        # full lidar dem image
        if self.lidar_dem_image is None: self.load_full_lidar_dem_image()

        image = self.lidar_dem_image[:,:,:]

        if full_gt_overlay:
            # If the ground truth image member variable is empty, then 
            # load the full ground truth image
            if self.gt_image is None: self.load_full_gt_image()

            classes = self.gt_image
            title = 'LiDAR Digital Elevation Model (DEM) image w/ ground truth overlay'
        elif test_gt_overlay or train_gt_overlay:
            # If the ground truth tiles image member variable is empty, 
            # then load the ground truth image tiles
            if self.gt_image_tiles is None: self.load_gt_image_tiles()

            # create a copy of ground truth image tiles
            gt_tiles = self.gt_image_tiles.copy()

            # Get tile dimensions
            tile_shape = gt_tiles[0].shape

            # Choose which set of tiles to set to zero values and 
            # set proper image title
            if train_gt_overlay: 
                tiles_to_remove = self.dataset_testing_subset
                title = 'LiDAR Digital Elevation Model (DEM) image w/ training ground truth overlay'
            else: 
                tiles_to_remove = self.dataset_training_subset
                title = 'LiDAR Digital Elevation Model (DEM) image w/ testing ground truth overlay'

            # Zero out tiles not in the desired subset
            for tile in tiles_to_remove:
                row, col = tile
                index = row * self.dataset_tiled_subset_cols + col
                gt_tiles[index] = np.zeros(tile_shape, dtype=np.uint8)
            
            # Create single ground truth image mask
            classes = self.merge_tiles(gt_tiles)

        else:
            classes = None
            title = 'LiDAR Digital Elevation Model (DEM) image'

        plt.close('all')

        view = spectral.imshow(image, 
                               source=image,
                               classes=classes,
                               figsize=size)
        if (full_gt_overlay 
            or test_gt_overlay 
            or train_gt_overlay): view.set_display_mode('overlay')
        
        view.set_title(title)

        plt.show(block=True)



    def load_full_lidar_ndsm_image(self, gsd=UH_2018_GT_GSD,
                                   use_void_filling_model = False,
                                   use_hybrid_model = True,
                                   thres=True, normalize=True):
        """
        Loads the full-size LiDAR normalized digital surface model 
        (NDSM) image for the University of Houston 2018 dataset sampled 
        at the specified GSD.
        """

        if use_hybrid_model: 
            dem_path = self.path_to_lidar_dem_b
            print('Loading full LiDAR NDSM image using hybrid DEM...')
        elif use_void_filling_model: 
            dem_path = self.path_to_lidar_dem_tli
            print('Loading full LiDAR NDSM image using bare-earth elevation w/ void filling DEM...')
        else:
            dem_path = self.path_to_lidar_dem_3msr
            print('Loading full LiDAR NDSM image using bare-earth elevation DEM...')

        # Check GSD parameter value
        if gsd <= 0: raise ValueError("'gsd' parameter must be greater than 0!")

        # Set the factor of GSD resampling
        resample_factor = self.gsd_lidar / float(gsd)

        # Get full path to dataset's LiDAR DSM image
        dsm_image_path = os.path.join(self.path_to_dataset_directory,
                                  self.path_to_lidar_dsm)
        dem_image_path = os.path.join(self.path_to_dataset_directory, dem_path)

        # Throw error if file paths do not exist
        if not os.path.isfile(dsm_image_path): raise FileNotFoundError(
            f'Path to UH2018 LiDAR DSM image is invalid! Path={dsm_image_path}')
        
        if not os.path.isfile(dem_image_path): raise FileNotFoundError(
            f'Path to UH2018 LiDAR DEM image is invalid! Path={dem_image_path}')
        
        # Create image variable
        self.lidar_ndsm_image = None

        # Open the LiDAR DSM and DEM files as dsm_src and dem_src
        with rasterio.open(dsm_image_path) as dsm_src, \
             rasterio.open(dem_image_path) as dem_src:

            # Set the shape of the resampled image
            dsm_out_shape=(dsm_src.count,
                           int(dsm_src.height * resample_factor), 
                           int(dsm_src.width * resample_factor))
            dem_out_shape=(dem_src.count,
                           int(dem_src.height * resample_factor), 
                           int(dem_src.width * resample_factor))

            # Read the images, resample it to the appropriate GSD, 
            # arrange the numpy array to be (rows, cols, bands), 
            # and remove unused bands
            dsm_image = np.moveaxis(dsm_src.read(
                                    out_shape=dsm_out_shape, 
                                    resampling=Resampling.nearest), 0, -1)
            dem_image = np.moveaxis(dem_src.read(
                                    out_shape=dem_out_shape, 
                                    resampling=Resampling.nearest), 0, -1)

            # Threshold the images so that any value over the threshold
            # is set to the image minimum
            if thres:
                dsm_image[dsm_image > self.lidar_dsm_thres] = dsm_image.min()
                dem_image[dem_image > self.lidar_dsm_thres] = dem_image.min()

            # NDSM is the difference between the DSM and the DEM
            self.lidar_ndsm_image = dsm_image - dem_image

            # Normalize each intensity band between 0.0 and 1.0
            if normalize:
                self.lidar_ndsm_image -= self.lidar_ndsm_image.min()
                self.lidar_ndsm_image /= self.lidar_ndsm_image.max()

        return self.lidar_ndsm_image



    def load_lidar_ndsm_image_tiles(self, gsd=UH_2018_GT_GSD, tile_list=None,
                                    use_void_filling_model = False,
                                    use_hybrid_model = True,
                                    thres=True, normalize=True):
        """
        Loads the University of Houston 2018 dataset's LiDAR normalized 
        digital surface model (NDSM) image as a set of tiles sampled at 
        a specified GSD. If no tile list is given, the whole image will 
        be loaded as tiles.
        """

        if use_hybrid_model: 
            dem_path = self.path_to_lidar_dem_b
            print('Loading full LiDAR NDSM image using hybrid DEM...')
        elif use_void_filling_model: 
            dem_path = self.path_to_lidar_dem_tli
            print('Loading full LiDAR NDSM image using bare-earth elevation w/ void filling DEM...')
        else:
            dem_path = self.path_to_lidar_dem_3msr
            print('Loading full LiDAR NDSM image using bare-earth elevation DEM...')

        # Check GSD parameter value
        if gsd <= 0: raise ValueError("'gsd' parameter must be greater than 0!")

        # Check tile_list parameter value
        if tile_list and not isinstance(tile_list, tuple): raise ValueError(
            "'tile_list' parameter should be a tuple of tuples!")

        # Set the factor of GSD resampling
        resample_factor = self.gsd_lidar / float(gsd)

        # Get full path to dataset's LiDAR DSM image
        dsm_image_path = os.path.join(self.path_to_dataset_directory,
                                  self.path_to_lidar_dsm)
        dem_image_path = os.path.join(self.path_to_dataset_directory, dem_path)

        # Throw error if file paths do not exist
        if not os.path.isfile(dsm_image_path): raise FileNotFoundError(
            f'Path to UH2018 LiDAR DSM image is invalid! Path={dsm_image_path}')
        
        if not os.path.isfile(dem_image_path): raise FileNotFoundError(
            f'Path to UH2018 LiDAR DEM image is invalid! Path={dem_image_path}')
        
        
        # Create image variable
        self.lidar_ndsm_image_tiles = []

        # Open the LiDAR DSM and DEM files as dsm_src and dem_src
        with rasterio.open(dsm_image_path) as dsm_src, \
             rasterio.open(dem_image_path) as dem_src:

            # Set the shape of the resampled image
            dsm_out_shape=(dsm_src.count,
                           int(dsm_src.height * resample_factor), 
                           int(dsm_src.width * resample_factor))
            dem_out_shape=(dem_src.count,
                           int(dem_src.height * resample_factor), 
                           int(dem_src.width * resample_factor))

            # Get the size of the tile windows
            dsm_tile_width = dsm_src.width / self.dataset_tiled_subset_cols
            dsm_tile_height = dsm_src.height / self.dataset_tiled_subset_rows
            dem_tile_width = dem_src.width / self.dataset_tiled_subset_cols
            dem_tile_height = dem_src.height / self.dataset_tiled_subset_rows

            # Read in the image data for each image tile
            for tile_row in range(0, self.dataset_tiled_subset_rows):
                for tile_column in range(0, self.dataset_tiled_subset_cols):

                    # If specified tiles are desired, then skip any
                    # tiles that do not match the tile_list parameter
                    if tile_list and (tile_row, tile_column) not in tile_list:
                        continue

                    # Set the tile windows to read from the image
                    dsm_window = Window(dsm_tile_width * tile_column,  
                                       dsm_tile_height * tile_row, 
                                       dsm_tile_width, dsm_tile_height)
                    dem_window = Window(dem_tile_width * tile_column,  
                                       dem_tile_height * tile_row, 
                                       dem_tile_width, dem_tile_height)

                    # Set the shapes of the resampled tiles
                    dsm_out_shape=(dsm_src.count,
                                   int(dsm_src.height * resample_factor), 
                                   int(dsm_src.width * resample_factor))
                    dem_out_shape=(dem_src.count,
                                   int(dem_src.height * resample_factor), 
                                   int(dem_src.width * resample_factor))

                    # Read the tile window from the image, resample it to
                    # the appropriate GSD, arrange the numpy array to be
                    # (rows, cols, bands), and remove unused bands
                    dsm_tile = np.moveaxis(dsm_src.read(
                                           window = dsm_window, 
                                           out_shape=dsm_out_shape, 
                                           resampling=Resampling.nearest), 0, -1)
                    
                    dem_tile = np.moveaxis(dem_src.read(
                                           window = dem_window, 
                                           out_shape=dem_out_shape, 
                                           resampling=Resampling.nearest), 0, -1)
                    
                    # Threshold the image tiles so that any value over 
                    # the threshold is set to the image minimum
                    if thres:
                        dsm_tile[dsm_tile > self.lidar_dsm_thres] = dsm_tile.min()
                        dem_tile[dem_tile > self.lidar_dsm_thres] = dem_tile.min()

                    # NDSM is the difference between the DSM and the DEM
                    ndsm_tile = dsm_tile - dem_tile

                    # Copy the tile to the tiles array
                    self.lidar_ndsm_image_tiles.append(ndsm_tile)

        # If no tiles were added to the tile list, then set image tiles
        # variable to 'None'
        if len(self.lidar_ndsm_image_tiles) == 0: self.lidar_ndsm_image_tiles = None
        else:
            # Turn list of numpy arrays into single numpy array
            self.lidar_ndsm_image_tiles = np.stack(self.lidar_ndsm_image_tiles)
            
            # Normalize each intensity band between 0.0 and 1.0
            if normalize:
                self.lidar_ndsm_image_tiles -= self.lidar_ndsm_image_tiles.min()
                self.lidar_ndsm_image_tiles /= self.lidar_ndsm_image_tiles.max()

        return self.lidar_ndsm_image_tiles



    def save_full_lidar_ndsm_image_array(self, path, file_name='full_lidar_ndsm_image.npy'):
        """
        Saves the numpy array of the full LiDAR ndsm image to a 
        file for faster loading in the future.
        """

        # If the lidar ndsm image member variable is empty, then load the full
        # lidar image
        if self.lidar_ndsm_image is None: self.load_full_lidar_ndsm_image()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.lidar_ndsm_image)
    


    def save_tiled_lidar_ndsm_image_array(self, path, file_name='tiled_lidar_ndsm_image.npy'):
        """
        Saves the numpy array of the tiled LiDAR ndsm image to 
        a file for faster loading in the future.
        """

        # If the lidar dsm image tile member variable is empty, then load all
        # lidar ndsm image tiles
        if self.lidar_ndsm_image_tiles is None: self.load_lidar_ndsm_image_tiles()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.lidar_ndsm_image_tiles)




    def load_full_lidar_ndsm_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset LiDAR normalized digital surface model image.
        """

        with open(file_path, 'rb') as infile:
            self.lidar_ndsm_image = np.load(infile)
    


    def load_tiled_lidar_ndsm_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset LiDAR normalized digital surface model image tiles.
        """

        with open(file_path, 'rb') as infile:
            self.lidar_ndsm_image_tiles = np.load(infile)



    def show_lidar_ndsm_image(self, size=(15,9), 
                            full_gt_overlay=False, 
                            train_gt_overlay=False,
                            test_gt_overlay=False):
        """
        Displays the LiDAR ndsm image with optional ground truth 
        overlay on the image.
        """

        # If the lidar ndsm image member variable is empty, then load the
        # full lidar ndsm image
        if self.lidar_ndsm_image is None: self.load_full_lidar_ndsm_image()

        image = self.lidar_ndsm_image[:,:,:]

        if full_gt_overlay:
            # If the ground truth image member variable is empty, then 
            # load the full ground truth image
            if self.gt_image is None: self.load_full_gt_image()

            classes = self.gt_image
            title = 'LiDAR Normalized Digital Surface Model (NDSM) image w/ ground truth overlay'
        elif test_gt_overlay or train_gt_overlay:
            # If the ground truth tiles image member variable is empty, 
            # then load the ground truth image tiles
            if self.gt_image_tiles is None: self.load_gt_image_tiles()

            # create a copy of ground truth image tiles
            gt_tiles = self.gt_image_tiles.copy()

            # Get tile dimensions
            tile_shape = gt_tiles[0].shape

            # Choose which set of tiles to set to zero values and 
            # set proper image title
            if train_gt_overlay: 
                tiles_to_remove = self.dataset_testing_subset
                title = 'LiDAR Normalized Digital Surface Model (NDSM) image w/ training ground truth overlay'
            else: 
                tiles_to_remove = self.dataset_training_subset
                title = 'LiDAR Normalized Digital Surface Model (NDSM) image w/ testing ground truth overlay'

            # Zero out tiles not in the desired subset
            for tile in tiles_to_remove:
                row, col = tile
                index = row * self.dataset_tiled_subset_cols + col
                gt_tiles[index] = np.zeros(tile_shape, dtype=np.uint8)
            
            # Create single ground truth image mask
            classes = self.merge_tiles(gt_tiles)

        else:
            classes = None
            title = 'LiDAR Normalized Digital Surface Model (NDSM) image'

        plt.close('all')

        view = spectral.imshow(image, 
                               source=image,
                               classes=classes,
                               figsize=size)
        if (full_gt_overlay 
            or test_gt_overlay 
            or train_gt_overlay): view.set_display_mode('overlay')
        
        view.set_title(title)

        plt.show(block=True)

    def load_full_vhr_image(self, gsd=UH_2018_GT_GSD,
                           thres=True, normalize=True):
        """
        Loads the full-size VHR RGB image for the University of
        Houston 2018 dataset sampled at the specified GSD.
        """
        print('Loading full VHR RGB image...')

        # VHR image can only be loaded as tiles since there's 14
        # images, so load tiles and then merge them to create full VHR
        # image
        self.vhr_image = self.merge_tiles(
            self.load_vhr_image_tiles(gsd=gsd, thres=thres, normalize=normalize))

        return self.gt_image
    
    def load_vhr_image_tiles(self, gsd=UH_2018_GT_GSD, tile_list=None,
                             thres=True, normalize=True):
        """
        Loads the University of Houston 2018 dataset's hyperspectral
        images as a set of tiles sampled at a specified GSD. If no tile
        list is given, the whole image will be loaded as tiles.
        """

        #TODO
        pass



    def save_full_vhr_image_array(self, path, file_name='full_vhr_image.npy'):
        """
        Saves the numpy array of the full vhr image to a file 
        for faster loading in the future.
        """

        # If the hs image member variable is empty, then load the full
        # vhr image
        if self.vhr_image is None: self.load_full_vhr_image()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.hs_image)
    


    def save_tiled_vhr_image_array(self, path, file_name='tiled_vhr_image.npy'):
        """
        Saves the numpy array of the tiled vhr image to a file 
        for faster loading in the future.
        """

        # If the hs image tile member variable is empty, then load all
        # vhr image tiles
        if self.vhr_image_tiles is None: self.load_vhr_image_tiles()

        with open(os.path.join(path, file_name), 'wb') as outfile:
            np.save(outfile, self.hs_image_tiles)




    def load_full_vhr_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset VHR image.
        """

        with open(file_path, 'rb') as infile:
            self.vhr_image = np.load(infile)
    


    def load_tiled_vhr_image_array(self, file_path):
        """
        Loads a saved numpy array for the University of Houston 2018
        dataset VHR image tiles.
        """

        with open(file_path, 'rb') as infile:
            self.vhr_image_tiles = np.load(infile)



    def show_vhr_image(self, size=(15,9), 
                      full_gt_overlay=False, 
                      train_gt_overlay=False,
                      test_gt_overlay=False):
        """
        Displays the vhr rgb image.
        """

        # If the hs image member variable is empty, then load the full
        # vhr rgb image
        if self.vhr_image is None: self.load_full_vhr_image()
        
        image = self.vhr_image[:,:,:]

        if full_gt_overlay:
            # If the gt image member variable is empty, then load the full
            # ground truth image
            if self.gt_image is None: self.load_full_gt_image()

            classes = self.gt_image
            title = 'VHR RGB image w/ ground truth overlay'
        elif test_gt_overlay or train_gt_overlay:
            # If the hs image member variable is empty, then load the
            # ground truth image tiles
            if self.gt_image_tiles is None: self.load_gt_image_tiles()

            # create a copy of ground truth image tiles
            gt_tiles = self.gt_image_tiles.copy()

            # Get tile dimensions
            tile_shape = gt_tiles[0].shape

            # Choose which set of tiles to set to zero values and 
            # set proper image title
            if train_gt_overlay: 
                tiles_to_remove = self.dataset_testing_subset
                title = 'VHR RGB image w/ training ground truth overlay'
            else: 
                tiles_to_remove = self.dataset_training_subset
                title = 'VHR RGB image w/ testing ground truth overlay'

            # Zero out tiles not in the desired subset
            for tile in tiles_to_remove:
                row, col = tile
                index = row * self.dataset_tiled_subset_cols + col
                gt_tiles[index] = np.zeros(tile_shape, dtype=np.uint8)
            
            # Create single ground truth image mask
            classes = self.merge_tiles(gt_tiles)

        else:
            classes = None
            title = 'VHR RGB image'

        plt.close('all')

        view = spectral.imshow(image, 
                               source=image,
                               classes=classes,
                               figsize=size)
        if (full_gt_overlay 
            or test_gt_overlay 
            or train_gt_overlay): view.set_display_mode('overlay')
        
        view.set_title(title)

        plt.show(block=True)

    def get_tile_indices(self, tile, row_offset=0, col_offset=0):
        """
        Returns the indices where there is a ground truth defined for a
        specific tile. Tile row and column offsets for the x and y
        values can also be defined.
        """

        # If the hs image member variable is empty, then load the
        # ground truth image tiles
        if self.gt_image_tiles is None: self.load_gt_image_tiles()

        tile_height, tile_width = self.gt_image_tiles[0].shape

        col_offset = col_offset * tile_width
        row_offset = row_offset * tile_height

        indices = []

        row, col = tile
        index = row * self.dataset_tiled_subset_cols + col
        img = self.gt_image_tiles[index]

        for r in range(tile_height):
            for c in range(tile_width):
                if img[r][c] > 0:
                    indices.append((r+row_offset,c+col_offset))
        
        return indices

    def get_train_test_split(self, flatten=False):
        """
        Returns the training and testing indicies of the dataset.
        """

        # If the hs image member variable is empty, then load the
        # ground truth image tiles
        if self.gt_image_tiles is None: self.load_gt_image_tiles()

        tile_height, tile_width = self.gt_image_tiles[0].shape

        image_width = tile_width * self.dataset_tiled_subset_cols

        train_indices = []
        test_indices = []

        for row in range(0, self.dataset_tiled_subset_rows):
            for col in range(0, self.dataset_tiled_subset_cols):
                tile = self.gt_image_tiles[row * self.dataset_tiled_subset_cols + col]
                for tr in range(tile_height):
                    for tc in range(tile_width):
                        r = tr + row*tile_height
                        c = tc + col*tile_width
                        if tile[tr][tc] > 0:
                            if flatten:
                                index = r*image_width + c
                            else:
                                index = (r,c)
                            if (row,col) in self.dataset_training_subset:
                                train_indices.append(index)
                            else:
                                test_indices.append(index)
        
        return train_indices, test_indices



### Main ###
if __name__ == "__main__":
    dataset = UH_2018_Dataset()

    HS_RGB = (4, 10, 23)

    # dataset.show_hs_image(#rgb_channels=HS_RGB, 
    #                       full_gt_overlay=False,
    #                       train_gt_overlay=False,
    #                       test_gt_overlay=False)

    dataset.visualize_hs_data_cube()

    dataset.get_gt_class_statistics(print_results=True)
    dataset.show_hs_image(rgb_channels=HS_RGB)
    dataset.show_lidar_ms_image()
    dataset.show_lidar_dsm_image()
    dataset.show_lidar_dem_image()
    dataset.show_lidar_ndsm_image()

    # hsi = dataset.load_full_hs_image(thres=False, normalize=False)
    # spectral.envi.save_image('2018_IEEE_GRSS_DFC_HSI_TR.hdr', hsi)

    # hsi = dataset.load_hs_image_tiles(tile_list=UH_2018_TRAINING_GT_TILES, thres=False, normalize=False)
    # hsi = dataset.merge_tiles(hsi, num_rows=1, num_cols=len(UH_2018_TRAINING_GT_TILES))
    # print(f'hsi shape: {hsi.shape}')
    # view = spectral.imshow(hsi, source=hsi, bands=HS_RGB, figsize=(15,9))
    # plt.show(block=True)
    # spectral.envi.save_image('2018_IEEE_GRSS_DFC_HSI_TR.hdr', hsi)

    # gt = dataset.load_full_gt_image()
    # with rasterio.open('2018_IEEE_GRSS_DFC_GT.tif',
    #                    'w',
    #                    driver='GTiff',
    #                    height=gt.shape[0],
    #                    width=gt.shape[1],
    #                    count=1,
    #                    dtype=gt.dtype,
    #                    nodata=None) as dst:
    #     dst.write(gt,1)

    # train_gt = dataset.load_full_gt_image(train_only=True)

    # print(f'train_gt shape: {train_gt.shape}')
    # with rasterio.open('2018_IEEE_GRSS_DFC_GT_TR.tif',
    #                    'w',
    #                    driver='GTiff',
    #                    height=train_gt.shape[0],
    #                    width=train_gt.shape[1],
    #                    count=1,
    #                    dtype=train_gt.dtype,
    #                    nodata=None) as dst:
    #     dst.write(train_gt,1)
