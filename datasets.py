#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset loading and setup module

This script sets up hyperspectral and datafusion data sets.

Author:  Christopher Good
Version: 1.0.0

Usage: datasets.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###
import math

### Other Library Imports ###
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import (
    Sequence,
    to_categorical, 
) 

### Local Imports ###
from grss_dfc_2018_uh import UH_2018_Dataset

### Global Variables ###
_grss_dfc_2018_dataset = None
_grss_dfc_2018_train_gt = None
_grss_dfc_2018_test_gt = None
_indian_pines_dataset = None
_pavia_center_dataset = None
_university_of_pavia_dataset = None

### Class Definitions ###

class HyperspectralDataset(Sequence):
    def __init__(self, data, gt, shuffle=True, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperspectralDataset, self).__init__()
        self.data = data
        self.label = gt
        self.shuffle = shuffle
        self.batch_size = hyperparams["batch_size"]
        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.center_pixel = hyperparams["center_pixel"]
        self.n_classes = hyperparams['n_classes']
        self.loss = hyperparams['loss']
        
        self.indices, self.labels = get_valid_indices(data, gt, **hyperparams)

        # Run epoch end function to initialize dataset
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, i):
        indices = self.indices[i*self.batch_size:(i+1)*self.batch_size]
        data_batch = []
        labels = []

        for index in indices:
            x, y = index
            x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
            x2, y2 = x1 + self.patch_size, y1 + self.patch_size

            data = self.data[x1:x2, y1:y2]
            label = self.label[x1:x2, y1:y2]

            # Copy the data into numpy arrays
            data = np.asarray(np.copy(data), dtype="float32")
            label = np.asarray(np.copy(label), dtype="int64")

            # Extract the center label if needed
            if self.center_pixel and self.patch_size > 1:
                label = label[self.patch_size // 2, self.patch_size // 2]
            # Remove unused dimensions when we work with invidual spectrums
            elif self.patch_size == 1:
                data = data[:, 0, 0]
                label = label[0, 0]

            # Add a fourth dimension for 3D CNN
            if self.patch_size > 1:
                # Make 4D data ((Batch x) Planes x Channels x Width x Height)
                # data = tf.expand_dims(data, 0)
                data = np.expand_dims(data, 0)

            if self.loss == 'categorical_crossentropy':
                label = to_categorical(label, num_classes = self.n_classes)

            data_batch.append(data)
            labels.append(label)

        data_batch = np.asarray(data_batch)
        labels = np.asarray(labels)

        # print(f'{i}>> data_batch shape: {data_batch.shape}')
        # print(f'{i}>> labels shape: {labels.shape}')

        return data_batch, labels

### Function Definitions ###

def get_valid_indices(data, gt, **hyperparams):
    patch_size = hyperparams["patch_size"]
    ignored_labels = set(hyperparams["ignored_labels"])
    supervision = hyperparams["supervision"]
    # Fully supervised : use all pixels with label not ignored
    if supervision == "full":
        mask = np.ones_like(gt)
        for l in ignored_labels:
            mask[gt == l] = 0
    # Semi-supervised : use all pixels, except padding
    elif supervision == "semi":
        mask = np.ones_like(gt)
    x_pos, y_pos = np.nonzero(mask)
    p = patch_size // 2
    indices = np.array(
        [
            (x, y)
            for x, y in zip(x_pos, y_pos)
            if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
        ]
    )
    labels = [gt[x, y] for x, y in indices]

    return indices, labels

def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)

    if mode == 'random':
       train_indices, test_indices = train_test_split(X, train_size=train_size, stratify=y)
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]
    elif mode == 'fixed':
       print(f'Sampling {mode} with train size = {train_size}')
       train_indices, test_indices = [], []
       for c in np.unique(gt):
           if c == 0:
              continue
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features

           train, test = train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
       train_indices = tuple([list(t) for t in zip(*train_indices)])
       test_indices = tuple([list(t) for t in zip(*test_indices)])
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / (first_half_count + second_half_count)
                    if ratio > 0.9 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError(f'{mode} sampling is not implemented yet.')
    return train_gt, test_gt

def load_grss_dfc_2018_uh_dataset(reload=False, **hyperparams):
    #TODO
    
    # Make sure the global variables are used
    global _grss_dfc_2018_dataset
    global _grss_dfc_2018_train_gt
    global _grss_dfc_2018_test_gt

    # To speed up processing, don't load or reload dataset unless
    # necessary
    if _grss_dfc_2018_dataset is None or reload:
        _grss_dfc_2018_dataset = UH_2018_Dataset()
        _grss_dfc_2018_train_gt = _grss_dfc_2018_dataset.load_full_gt_image(train_only=True)
        _grss_dfc_2018_test_gt = _grss_dfc_2018_dataset.load_full_gt_image(test_only=True)

    dataset = _grss_dfc_2018_dataset
    train_gt = _grss_dfc_2018_train_gt
    test_gt = _grss_dfc_2018_test_gt

    data = None

    # Check to see if hyperspectral data is being used
    if hyperparams['use_hs_data'] or hyperparams['use_all_data']:
        if dataset.hs_image is None:
            hs_data = dataset.load_full_hs_image()
        else:
            hs_data = dataset.hs_image
        print(f'{dataset.name} hs_data shape: {hs_data.shape}')
        if data is None:
            data = np.copy(hs_data)
        else:
            data = np.dstack((data, hs_data))

    # Check to see if lidar multispectral intensity data is being used
    if hyperparams['use_lidar_ms_data'] or hyperparams['use_all_data']:
        if dataset.lidar_ms_image is None:
            lidar_ms_data = dataset.load_full_lidar_ms_image()
        else:
            lidar_ms_data = dataset.lidar_ms_image
        print(f'{dataset.name} lidar_ms_data shape: {lidar_ms_data.shape}')
        if data is None:
            data = np.copy(lidar_ms_data)
        else:
            data = np.dstack((data, lidar_ms_data))

    # Check to see if lidar normalized digital surface model data is
    # being used
    if hyperparams['use_lidar_ndsm_data'] or hyperparams['use_all_data']:
        if dataset.lidar_ndsm_image is None:
            lidar_ndsm_data = dataset.load_full_lidar_ndsm_image()
        else:
            lidar_ndsm_data = dataset.lidar_ndsm_image
        print(f'{dataset.name} lidar_ndsm_data shape: {lidar_ndsm_data.shape}')
        if data is None:
            data = np.copy(lidar_ndsm_data)
        else:
            data = np.dstack((data, lidar_ndsm_data))

    # Check to see if very high resolution RGB image data is being used
    if hyperparams['use_vhr_data'] or hyperparams['use_all_data']:
        if dataset.vhr_image is None:
            vhr_data = dataset.load_full_vhr_image()
        else:
            vhr_data = dataset.vhr_image
        print(f'{dataset.name} vhr_data shape: {vhr_data.shape}')
        if not data:
            data = np.copy(vhr_data)
        else:
            data = np.dstack((data, vhr_data))
    
    # Verify that some data was loaded
    if data is not None:
        print(f'{dataset.name} full dataset shape: {data.shape}')
    else:
        print('No data was loaded! Training cancelled...')
        return


    print(f'{dataset.name} train_gt shape: {train_gt.shape}')
    print(f'{dataset.name} test_gt shape: {test_gt.shape}')

    dataset_info = {
        'name': dataset.name,
        'num_classes': dataset.gt_num_classes,
        'ignored_labels': dataset.gt_ignored_labels,
        'class_labels': dataset.gt_class_label_list,
        'label_mapping': dataset.gt_class_value_mapping,
    }

    return data, train_gt, test_gt, dataset_info

def load_indian_pines_dataset(**hyperparams):
    #TODO

    data = None
    train_gt = None
    test_gt = None

    labels = [
            'Undefined',
            'Alfalfa',
            'Corn-notill',
            'Corn-mintill',
            'Corn',
            'Grass-pasture',
            'Grass-trees',
            'Grass-pasture-mowed',
            'Hay-windrowed',
            'Oats',
            'Soybean-notill',
            'Soybean-mintill',
            'Soybean-clean',
            'Wheat',
            'Woods',
            'Buildings-Grass-Trees-Drives',
            'Stone-Steel-Towers',
        ]

    dataset_info = {
        'name': 'Indian Pines',
        'num_classes': len(labels),
        'ignored_labels': [0],
        'class_labels': labels,
        'label_mapping': {index: label for index, label in enumerate(labels)},
    }

    return data, train_gt, test_gt, dataset_info

def load_pavia_center_dataset(**hyperparams):
    #TODO
    
    data = None
    train_gt = None
    test_gt = None

    labels = [
            'Undefined',
            'Water',
            'Trees',
            'Asphalt',
            'Self-Blocking Bricks',
            'Bitumen',
            'Tiles',
            'Shadows',
            'Meadows',
            'Bare Soil',
        ]

    dataset_info = {
        'name': 'University of Pavia',
        'num_classes': len(labels),
        'ignored_labels': [0],
        'class_labels': labels,
        'label_mapping': {index: label for index, label in enumerate(labels)},
    }

    return data, train_gt, test_gt, dataset_info

def load_university_of_pavia_dataset(**hyperparams):
    #TODO
    
    data = None
    train_gt = None
    test_gt = None

    labels = [
            'Undefined',
            'Asphalt',
            'Meadows',
            'Gravel',
            'Trees',
            'Painted metal sheets',
            'Bare Soil',
            'Bitumen',
            'Self-Blocking Bricks',
            'Shadows',
        ]

    dataset_info = {
        'name': 'University of Pavia',
        'num_classes': len(labels),
        'ignored_labels': [0],
        'class_labels': labels,
        'label_mapping': {index: label for index, label in enumerate(labels)},
    }

    return data, train_gt, test_gt, dataset_info

def create_datasets(data, train_gt, test_gt, **hyperparams):
    #TODO

    patch_size = hyperparams['patch_size']  # N in NxN patch per sample
    train_split = hyperparams['train_split']    # training percent in val/train split

    # Set pad length per dimension
    pad = patch_size // 2

    # Pad only first two dimensions
    data = np.pad(data, [(pad,), (pad,), (0,)], mode='constant')
    train_gt = np.pad(train_gt, [(pad,), (pad,)], mode='constant')
    test_gt = np.pad(test_gt, [(pad,), (pad,)], mode='constant')

    # Show updated padded dataset shapes
    print(f'padded data shape: {data.shape}')
    print(f'padded train_gt shape: {train_gt.shape}')
    print(f'padded test_gt shape: {test_gt.shape}')

    # Create validation dataset from training set
    # train_gt, val_gt = sample_gt(train_gt, train_split, mode='random')
    train_gt, val_gt = sample_gt(train_gt, train_split, mode='fixed')

    train_dataset = HyperspectralDataset(data, train_gt, **hyperparams)
    val_dataset = HyperspectralDataset(data, val_gt, **hyperparams)
    test_dataset = HyperspectralDataset(data, test_gt, shuffle=False, **hyperparams)
    true_test = np.array(test_dataset.labels)

    return train_dataset, val_dataset, test_dataset, true_test