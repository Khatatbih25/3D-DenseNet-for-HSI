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
#
#
# --- Hyper-Spectral ---
# HSI sensor: ITRES CASI 1500
# Number of bands: 48
# Spectral range: 380-1050nm
# GSD: 1-m GSD
#
########################################################################


########################################################################
# Imports
########################################################################
import wx
import spectral
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import rasterio
from rasterio.enums import Resampling

########################################################################
# Constants
########################################################################
HOUSTON_DATASET_PATH = 'datasets/grss_dfc_2018/'
TRAINING_GT_IMAGE_PATH = HOUSTON_DATASET_PATH + 'TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif'
HSI_IMAGE_PATH = HOUSTON_DATASET_PATH + 'FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix'
LIDAR_DSM_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/DSM_C12/UH17c_GEF051.tif'
LIDAR_DEM_3MSR_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/DEM_C123_3msr/UH17_GEG051.tif'
LIDAR_DEM_TLI_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/DEM_C123_TLI/UH17_GEG05.tif'
LIDAR_DEM_B_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/DEM+B_C123/UH17_GEM051.tif'
LIDAR_INTENSITY_1550NM_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/Intensity_C1/UH17_GI1F051.tif'
LIDAR_INTENSITY_1064NM_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/Intensity_C2/UH17_GI2F051.tif'
LIDAR_INTENSITY_532NM_PATH = HOUSTON_DATASET_PATH + 'Lidar GeoTiff Rasters/Intensity_C3/UH17_GI3F051.tif'


########################################################################
# Functions
########################################################################
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

def open_hs_rasterio():

    # Open the training HSI Envi file as src and ground truth file as gt_src
    with rasterio.open(HSI_IMAGE_PATH, format='ENVI') as src, \
         rasterio.open(TRAINING_GT_IMAGE_PATH) as gt_src:
        
        # Output training image data
        print(f"bands: {src.count}")
        print(f"width: {src.width}")
        print(f"height: {src.height}")

        # Read in ground truth image
        gt = gt_src.read(1)

        # Read in training image, resampled to ground truth image size
        # using nearest-neighbor method
        resampled_img = np.moveaxis(src.read(out_shape=(src.count, gt_src.height, gt_src.width), resampling=Resampling.nearest), 0, -1)[:,:,:-2]
        
        # Print out resampled image and ground truth image shapes
        print(f"resampled img shape: {resampled_img.shape}")
        print(f"gt shape: {gt.shape}")
        
        # Show resampled image
        spectral.imshow(classes=gt)
        view = spectral.imshow(resampled_img, source = resampled_img, classes=gt, figsize=(15, 9))
        #view.set_display_mode('overlay')
        plt.pause(0)

        # Show image cube
        spectral.view_cube(resampled_img, size=(1200, 900))

def open_hs_spectral():
    # Import hyperpectral image
    full_hs_img = spectral.envi.open('datasets/grss_dfc_2018/FullHSIDataset/20170218_UH_CASI_S4_NAD83.hdr',
                    'datasets/grss_dfc_2018/FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix')


    # Import ground truth
    gt = spectral.open_image('datasets/grss_dfc_2018/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.hdr').read_band(0)
    #gt = imageio.imread('datasets/grss_dfc_2018/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif')

    print(full_hs_img)
    #print(gt)
    print(f"full image shape: {full_hs_img.shape}")
    #print(f"gt image shape: {gt.shape}")

    # Remove edge bands from hs image
    #corrected_hs_img = full_hs_img.load()[:,:,:-2]

    # Print HS image data
    #print(corrected_hs_img)

    # Initialize settings
    plot_width = 15 # in inches
    plot_height = 9 # in inches

    # Show HSI as 2-D image w/ spectrum graph when double-clicked
    #spectral.imshow(classes=gt)
    spectral.imshow(full_hs_img[:,:,:], source = full_hs_img[:,:,:], figsize=(plot_width, plot_height))
    #spectral.imshow(corrected_hs_img, source = corrected_hs_img, figsize=(plot_width, plot_height))

    # See full 3-D HSI cube
    #spectral.view_cube(full_hs_img, size=(1200, 900))

    #spectral.view(corrected_hs_img)

    #pc = spectral.principal_components(full_hs_img)
    #data = pc.transform(full_hs_img)
    #spectral.view_nd(corrected_hs_img[:,:,:], classes = gt)


    #spectral.ImageView(corrected_hs_img, source=corrected_hs_img).show()

    # Prevent plots from closing immediately
    plt.pause(0)


########################################################################
# Main
########################################################################
if __name__ == "__main__":
    # Setup wxApp
    app = wx.App(False)

    open_hs_rasterio()
    #open_lidar_rasterio()

    # Prevent apps from closing immediately
    app.MainLoop()