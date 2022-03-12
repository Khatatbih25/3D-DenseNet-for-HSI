########################################################################
# ABOUT FILE
########################################################################
# Python script to allow visualization of hyperpectral, lidar, and VHR
# RGB imagery for the Houston 2018 dataset.
########################################################################

########################################################################
# NOTES
########################################################################
#
# --- LIDAR ---
# Lidar sensor: Optech Titan MW (14SEN/CON340) w/ integrated camera
#               ~ This is a multispectral lidar sensor
# Lidar channel 1 = 1550nm wavelength (near infrared)
# Lidar channel 2 = 1064nm wavelength (near infrared)
# Lidar channel 3 = 532nm wavelength (green)
# GSD: 0.5-m ground sampling distance (this is for MS-LiDAR point cloud
#      data, digital surface model (DSM), and rasterized instensity)
# # - DEM_C123_3msr is a bare-earth digital elevation model (DEM) generated
#    from returns classified as ground from all three sensors
# - DEM_C123_TLI is a bare-earth DEM with void filling for manmade structures
# - DEM+B_C123 is a hybrid ground and building DEM, generated from returns
#    that were classified as coming from buildings and the ground by all
#    titan sensors
#
# --- Hyper-Spectral ---
# HSI sensor: ITRES CASI 1500
# Number of bands: 48
# Spectral range: 380-1050nm
# GSD: 1-m GSD
#
#
#
# TFW file layout:
# Line 1: Pixel size in the x-direction in map units (GSD)
# Line 2: rotation about y-axis
# Line 3: rotation about x-axis
# Line 4: pixel size in the y direction in map units (GSD)
# Line 5: x-coordinate of the upper left corner of the image
# Line 6: y-coordinate of the upper left corner of the image
########################################################################


########################################################################
# Imports
########################################################################
import wx
import spectral
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import pandas as pd
import os.path


########################################################################
# Constants
########################################################################
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

# A list of the wavelength values for each of the hyperspectal band
# channels
HS_BAND_WAVELENGTHS = [
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
BAND_RGB = [
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

# Number of defined classes in the ground truth image
NUMBER_OF_CLASSES = 20

# Map of classes where the key is the value of the pixel in the
# ground truth image
CLASS_MAP = {
    0:  'Undefined',
    1:  'Healthy grass',
    2:  'Stressed grass',
    3:  'Artificial turf',
    4:  'Evergreen trees',
    5:  'Deciduous trees',
    6:  'Bare earth',
    7:  'Water',
    8:  'Residential buildings',
    9:  'Non-residential buildings',
    10: 'Roads',
    11: 'Sidewalks',
    12: 'Crosswalks',
    13: 'Major thoroughfares',
    14: 'Highways',
    15: 'Railways',
    16: 'Paved parking lots',
    17: 'Unpaved parking lots',
    18: 'Cars',
    19: 'Trains',
    20: 'Stadium seats',
}
    
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
########################################################################
# Functions
########################################################################


def load_vhr_images():

    # Initialize list for tiles
    vhr_tiles = []

    # Set the factor of GSD resampling 1/factor for 1m to 0.5m
    resample_factor = VHR_GSD / GT_GSD

    for row, tile_paths in enumerate(VHR_IMAGE_PATHS):
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

def load_hs_images():

    # Initialize list for tiles
    hs_tiles = []
    
    # Set the factor of GSD resampling 1/factor for 1m to 0.5m
    resample_factor = HS_GSD / GT_GSD

    # Open the training HSI Envi file as src
    with rasterio.open(HSI_IMAGE_PATH, format='ENVI') as src:
        # Get the size of the tile windows
        tile_width = src.width / COLUMN_TILES
        tile_height = src.height / ROW_TILES

        # Read in the image data for each image tile
        for tile_row in range(0, ROW_TILES):
            hs_tiles.append([])
            for tile_column in range(0, COLUMN_TILES):

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

def load_lidar_intensity_cube():
    # Initialize list for tiles
    lidar_intensity_tiles = []

    # Open the training HSI Envi file as src
    with rasterio.open(LIDAR_INTENSITY_1550NM_PATH) as c1_src, \
         rasterio.open(LIDAR_INTENSITY_1064NM_PATH) as c2_src, \
         rasterio.open(LIDAR_INTENSITY_532NM_PATH) as c3_src:

        # Get the size of the tile windows (all should be same)
        tile_width = c1_src.width / COLUMN_TILES
        tile_height = c1_src.height / ROW_TILES

        # Read in the image data for each image tile
        for tile_row in range(0, ROW_TILES):
            lidar_intensity_tiles.append([])
            for tile_column in range(0, COLUMN_TILES):

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

def load_ground_truth():
    # Initialize list for tiles
    gt_tiles = []

    with rasterio.open(TRAINING_GT_IMAGE_PATH) as train_src, \
         rasterio.open(TESTING_GT_IMAGE_PATH) as test_src:

        # Get the size of the tile windows (use full size test image)
        tile_width = test_src.width / COLUMN_TILES
        tile_height = test_src.height / ROW_TILES

            # Read in the image data for each image tile
        for tile_row in range(0, ROW_TILES):
            gt_tiles.append([])
            for tile_column in range(0, COLUMN_TILES):

                # Check to see if current tile is one of the training
                # ground truth tiles
                if (tile_row, tile_column) in TRAINING_GT_TILES:

                    offset_row, offset_column = TRAINING_GT_TILE_OFFSETS[
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

def get_class_statistics(img_dict):

    # Setup variables
    num_samples_per_class = np.zeros(NUMBER_OF_CLASSES+1)
    avg_class_wl_intensity = np.zeros((NUMBER_OF_CLASSES+1, NUM_HS_BANDS))
    labeled_pixels = []
    pixel_class_list = [[] for i in range(NUMBER_OF_CLASSES+1)]

    # Plotting variables
    band_list = [x for x in range(NUM_HS_BANDS)]
    band_labels = [x.split(' ')[0] for x in HS_BAND_WAVELENGTHS]

    # Get image size
    rows, columns = img_dict['gt'].shape[0:2]

    # Find labeled pixels
    for r in range(rows):
        for c in range(columns):
            # Verify that pixel has a defined class label before adding
            # it to labeled pixel list
            if img_dict['gt'][r][c] > 0:
                labeled_pixels.append((r, c))

            # Append pixel to corresponding class pixel list
            pixel_class_list[img_dict['gt'][r][c]].append((r, c))

    # Get statistics for each class
    for label, pixel_list in enumerate(pixel_class_list):
        if label > 0:
            # Count samples per class
            num_samples_per_class[label] = len(pixel_list)

            # Add up the wavelength reflectance intensities for each
            # pixel of current class label
            for r, c in pixel_list:
                avg_class_wl_intensity[label] += img_dict['hs'][r][c]
            
            # Divide each wavelength sum by the number of samples per
            # current class label to get average wavelengths for this
            # class
            avg_class_wl_intensity[label] /= num_samples_per_class[label]

            # Set up average wavelength bar graph plot
            plt.figure(figsize=(12,9))
            plt.bar(band_list, avg_class_wl_intensity[label], color=BAND_RGB)
            plt.grid(color='grey', linestyle='-', linewidth=1, axis='y')
            plt.xticks(band_list, band_labels, rotation='vertical')
            plt.xlabel('Wavelength')
            plt.ylim(0, 8000)
            plt.ylabel('Intensity')
            plt.title(f'Average wavelength for {CLASS_MAP[label]}')
            plt.tight_layout()

            # Create filename for bar plot image
            plot_file_name = ('analysis/class_data/avg_wavelength__'
                    + CLASS_MAP[label].replace(' ', '_') + '.png')

            # Save bar plot image
            print(f'>>> Saving avg wavelength plot {plot_file_name}...')
            plt.savefig(plot_file_name, bbox_inches='tight')

            # plt.show()

            # Clear plot data for next plot
            plt.clf()
    
    # Create CSV file containing class sample numbers and average band
    # wavelength data
    csv = pd.DataFrame(avg_class_wl_intensity[1:], columns=band_labels)
    csv.insert(0, 'Classes', CLASS_LIST[1:])
    csv.insert(1, 'Total Samples', num_samples_per_class[1:])
    csv.to_csv('analysis/average_wavelengths.csv')

    # Print number of samples per class
    print('Samples per class:')
    for i in range(1, NUMBER_OF_CLASSES+1):
        print(f'{CLASS_MAP[i]}: {num_samples_per_class[i]}')
    print()

def show_3d_cube(array3d):
    # Setup wxApp
    app = wx.App(False)

    spectral.view_cube(array3d, size=(1200, 900))

    # Prevent apps from closing immediately
    app.MainLoop()

def open_lidar_rasterio():
    with rasterio.open(TRAINING_GT_IMAGE_PATH) as gt_src, \
         rasterio.open(LIDAR_DSM_PATH) as lidar_dsm_src, \
         rasterio.open(LIDAR_DEM_3MSR_PATH) as lidar_dem_3msr_src, \
         rasterio.open(LIDAR_DEM_TLI_PATH) as lidar_dem_tli_src, \
         rasterio.open(LIDAR_DEM_B_PATH) as lidar_dem_b_src:

        print('--- Data ---')

        # Print Ground Truth Data
        print('Ground Truth:')
        print(f'name: {gt_src.name}')
        print(f'bands: {gt_src.count}')
        print(f'width: {gt_src.width}')
        print(f'height: {gt_src.height}')
        print(f'bounds: {gt_src.bounds}')
        print(f'crs: {gt_src.crs}')
        print()

        # Print Lidar Digital Surface Model Channels 1 & 2 Data
        print('Lidar Digital Surface Model (DSM) [Channels 1 & 2]:')
        print(f'name: {lidar_dsm_src.name}')
        print(f'bands: {lidar_dsm_src.count}')
        print(f'width: {lidar_dsm_src.width}')
        print(f'height: {lidar_dsm_src.height}')
        print(f'bounds: {lidar_dsm_src.bounds}')
        print(f'crs: {lidar_dsm_src.crs}')
        print()

        # Print Lidar Digital Elevation Model 3MSR
        print('Lidar Digital Elevation Model (DEM) 3MSR:')
        print(f'name: {lidar_dem_3msr_src.name}')
        print(f'bands: {lidar_dem_3msr_src.count}')
        print(f'width: {lidar_dem_3msr_src.width}')
        print(f'height: {lidar_dem_3msr_src.height}')
        print(f'bounds: {lidar_dem_3msr_src.bounds}')
        print(f'crs: {lidar_dem_3msr_src.crs}')
        print()

        # Print Lidar Digital Elevation Model TLI
        print('Lidar Digital Elevation Model (DEM) TLI:')
        print(f'name: {lidar_dem_tli_src.name}')
        print(f'bands: {lidar_dem_tli_src.count}')
        print(f'width: {lidar_dem_tli_src.width}')
        print(f'height: {lidar_dem_tli_src.height}')
        print(f'bounds: {lidar_dem_tli_src.bounds}')
        print(f'crs: {lidar_dem_tli_src.crs}')
        print()

        # Print Lidar Digital Elevation Model B
        print('Lidar Digital Elevation Model (DEM) B:')
        print(f'name: {lidar_dem_b_src.name}')
        print(f'bands: {lidar_dem_b_src.count}')
        print(f'width: {lidar_dem_b_src.width}')
        print(f'height: {lidar_dem_b_src.height}')
        print(f'bounds: {lidar_dem_b_src.bounds}')
        print(f'crs: {lidar_dem_b_src.crs}')
        print()

        # Read in ground truth image
        gt = gt_src.read(1)

        # Read in Lidar Digital Surface Model Channels 1 & 2 image data
        # and move band dimension from first to last
        lidar_dsm_img = np.moveaxis(lidar_dsm_src.read(), 0, -1)

        # Read in Lidar Digital Elevation Model 3MSR image data and move
        # band dimension from first to last
        lidar_dem_3msr_img = np.moveaxis(lidar_dem_3msr_src.read(), 0, -1)

        # Read in Lidar Digital Elevation Model TLI image data and move
        # band dimension from first to last
        lidar_dem_tli_img = np.moveaxis(lidar_dem_tli_src.read(), 0, -1)

        # Read in Lidar Digital Elevation Model B image data and move
        # band dimension from first to last
        lidar_dem_b_img = np.moveaxis(lidar_dem_b_src.read(), 0, -1)

        # Create plots
        spectral.imshow(classes=gt, title='Ground Truth')
        spectral.imshow(lidar_dsm_img, source=lidar_dsm_img, title='Lidar DSM (Channels 1 & 2)', figsize=(12, 9))
        spectral.imshow(lidar_dem_3msr_img, source=lidar_dem_3msr_img, title='Lidar DEM 3MSR', figsize=(12, 9))
        spectral.imshow(lidar_dem_tli_img, source=lidar_dem_tli_img, title='Lidar DEM TLI', figsize=(12, 9))
        spectral.imshow(lidar_dem_b_img, source=lidar_dem_b_img, title='Lidar DEM B', figsize=(12, 9))

        # Pause plot so that it can be examined
        plt.pause(0)


########################################################################
# Main
########################################################################
if __name__ == "__main__":

    # Initialize variables
    tiled_dataset = {}

    # Load hyperspectral image data
    if os.path.exists('preprocessed_datasets/hs_tiles.npy'):
        with open('preprocessed_datasets/hs_tiles.npy', 'rb') as infile:
            print('Loading hyperspectral tile array...')
            tiled_dataset['hs'] = np.load(infile)
    else:
        print('Loading hyperspectral image...')
        tiled_dataset['hs'] = load_hs_images()
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
        tiled_dataset['lidar'] = load_lidar_intensity_cube()
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
        tiled_dataset['vhr'] = load_vhr_images()
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
        tiled_dataset['gt'] = load_ground_truth()
        print('>>> Saving ground truth tile array...')
        with open('preprocessed_datasets/gt_tiles.npy', 'wb') as outfile:
            np.save(outfile, tiled_dataset['gt'])


    # print('Creating full image dataset dictionary...')
    # full_image_dataset = {
    #     'hs': merge_tiles(hs_tiles),
    #     'lidar': merge_tiles(lidar_intensity_tiles),
    #     'vhr': merge_tiles(vhr_tiles),
    #     'gt': merge_tiles(gt_tiles),
    # }

    # Set up training set from full dataset
    print('Creating training set dictionary...')
    training_set = {
        'hs': merge_tiles(tiled_dataset['hs'][[1],1:5,...]),
        'lidar': merge_tiles(tiled_dataset['lidar'][[1], 1:5, ...]),
        'vhr': merge_tiles(tiled_dataset['vhr'][[1], 1:5, ...]),
        'gt': merge_tiles(tiled_dataset['gt'][[1],1:5,...]),
    }

    # Get statistics on image data
    print('Generating class statistics...')
    get_class_statistics(training_set)

    # Create a full hyperspectral image to get grayscale of bands
    print('Creating full hyperspectral image...')
    full_hs_image = merge_tiles(tiled_dataset['hs'])

    # Create grayscale images of each spectral band
    print('Creating and saving grayscale images for hyperspectral bands...')
    for bands in range(0, full_hs_image.shape[-1]):
        image_file_name = (f'analysis/hs_greyscale_intensity/hs_band_{bands+1}__' 
            + HS_BAND_WAVELENGTHS[bands].split(' ')[0].replace('.','pt')
            + '.png')
        print(f'>>> Saving band#{bands} ~{HS_BAND_WAVELENGTHS[bands]}~ to {image_file_name}')
        spectral.save_rgb(image_file_name, full_hs_image, [bands], format='png')

    # Clear all prior figures
    plt.close('all')

    # View hyperspectral training set overlaid with ground truth
    view = spectral.imshow(training_set['hs'], 
                           source = training_set['hs'], 
                           classes=training_set['gt'], 
                           figsize=(15, 9))
    view.set_title('Hyperspectral Training Set w/ GT Overlay')
    view.set_display_mode('overlay')

    # View lidar intensity training set overlaid with ground truth
    view = spectral.imshow(training_set['lidar'], 
                           source = training_set['lidar'], 
                           classes=training_set['gt'], 
                           figsize=(15, 9))
    view.set_display_mode('overlay')
    view.set_title('LiDAR Intensity Training Set w/ GT Overlay')
    print(view)

    # View VHR RGB image training set overlaid with ground truth
    view = spectral.imshow(training_set['vhr'], 
                           source = training_set['vhr'], 
                           classes=training_set['gt'], 
                           figsize=(15, 9))
    view.set_title('VHR RGB Training Set w/ GT Overlay')
    view.set_display_mode('overlay')
    print(view)

    plt.show(block=True)

    show_3d_cube(training_set['hs'])
