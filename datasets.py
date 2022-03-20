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
import tensorflow as tf
from tensorflow.keras.utils import (
    Sequence,
    to_categorical, 
) 

### Local Imports ###
from grss_dfc_2018_uh import UH_2018_Dataset

### Class Definitions ###
# class HyperspectralDataset(Sequence):
#     def __init__(self, data, gt, shuffle=True, **hyperparams):
#         """
#         Args:
#             data: 3D hyperspectral image
#             gt: 2D array of labels
#             patch_size: int, size of the spatial neighbourhood
#             center_pixel: bool, set to True to consider only the label of the
#                           center pixel
#             data_augmentation: bool, set to True to perform random flips
#             supervision: 'full' or 'semi' supervised algorithms
#         """
#         super(HyperspectralDataset, self).__init__()
#         self.data = data
#         self.gt = gt
#         self.shuffle = shuffle
#         self.batch_size = hyperparams["batch_size"]
#         self.patch_size = hyperparams["patch_size"]
#         self.ignored_labels = set(hyperparams["ignored_labels"])
#         self.center_pixel = hyperparams["center_pixel"]
#         self.num_classes = hyperparams['n_classes']
#         self.loss = hyperparams['loss']
        
#         self.indices, self.labels = get_valid_indices(data, gt, 
#             self.patch_size, self.ignored_labels, hyperparams['supervision'])

#         # Run epoch end function to initialize dataset
#         self.on_epoch_end()

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)

#     def __len__(self):
#         return math.ceil(len(self.indices) / self.batch_size)

#     def __getitem__(self, i):
#         batch_data = []
#         batch_labels = []

#         for item in range(i*self.batch_size,(i+1)*self.batch_size):
#             if item >= len(self.indices): break
#             index = tuple(self.indices[item])
#             data = get_data_patch(self.data, index, self.patch_size)
#             label = self.gt[index]
#             if self.loss == 'categorical_crossentropy':
#                 label = to_categorical(label, num_classes = self.num_classes)

#             batch_data.append(data)
#             batch_labels.append(label)

#         batch_data = tf.convert_to_tensor(batch_data)
#         batch_labels = tf.convert_to_tensor(batch_labels)

#         return batch_data, batch_labels

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
        # super(HyperspectralDataset, self).__init__()
        self.data = data
        self.gt = gt
        self.shuffle = shuffle
        self.batch_size = hyperparams["batch_size"]
        self.patch_size = hyperparams["patch_size"]
        self.supervision = hyperparams['supervision']
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.num_classes = hyperparams['n_classes']
        self.loss = hyperparams['loss']
        
        if self.supervision == "full":
            mask = np.ones_like(gt)
            for label in self.ignored_labels:
                mask[gt == label] = 0
        # Semi-supervised : use all pixels, except padding
        elif self.supervision == "semi":
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        num_neighbors = self.patch_size // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x > num_neighbors 
                    and x < data.shape[0] - num_neighbors 
                    and y > num_neighbors 
                    and y < data.shape[1] - num_neighbors
            ]
        )

        self.labels = np.array([gt[x, y] for x, y in self.indices])

        # Run epoch end function to initialize dataset
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, i):
        batch_data = []
        batch_labels = []

        # Get all items in batch
        for item in range(i*self.batch_size,(i+1)*self.batch_size):

            # Make sure not to look for item id greater than number of
            # indices
            if item >= len(self.indices): break

            # Get index tuple from indices
            index = tuple(self.indices[item])

            # Get data patch for the index
            data = self.__get_data_patch(self.data, index, self.patch_size)

            # Get label for the patch
            label = self.gt[index]

            # If categorical cross-entropy, make sure labels are one-hot
            # encoded
            if self.loss == 'categorical_crossentropy':
                label = to_categorical(label, num_classes = self.num_classes)

            # Add data to lists
            batch_data.append(data)
            batch_labels.append(label)

        batch_data = tf.convert_to_tensor(batch_data)
        batch_labels = tf.convert_to_tensor(batch_labels)

        return batch_data, batch_labels

    @staticmethod
    def __get_data_patch(data, index, patch_size):
        x, y = index
        x1 = x - patch_size // 2    # Leftmost edge of patch
        y1 = y - patch_size // 2    # Topmost edge of patch
        x2 = x1 + patch_size        # Rightmost edge of patch
        y2 = y1 + patch_size        # Bottommost edge of patch

        patch = data[x1:x2, y1:y2]

        # Copy the data into numpy arrays
        # patch = np.asarray(np.copy(patch), dtype="float32")
        patch = tf.convert_to_tensor(patch, dtype="float32")

        if patch_size == 1:
            patch = patch[:, 0, 0]

        # Add a fourth dimension for 3D CNN
        if patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            # patch = np.expand_dims(patch, 0)
            patch = tf.expand_dims(patch, 0)
        
        return patch


### Function Definitions ###

def hs_dataset_generator(data, gt, shuffle=True, **hyperparams):

    patch_size = hyperparams['patch_size']
    ignored_labels = hyperparams['ignored_labels']
    num_classes = hyperparams['n_classes']
    supervision = hyperparams['supervision']
    batch_size = hyperparams['batch_size']
    loss = hyperparams['loss']

    indices, labels = get_valid_indices(data, gt, 
                                        patch_size=patch_size,
                                        ignored_labels=ignored_labels,
                                        supervision=supervision)
    
    if shuffle:
            np.random.shuffle(indices)

    batch = 0
    while batch*batch_size < len(indices):
        batch_data = []
        batch_labels = []
        for i in range(batch*batch_size,(batch+1)*batch_size):
            if i >= len(indices): break
            index = tuple(indices[i])
            label = gt[index]
            if loss == 'categorical_crossentropy':
                label = to_categorical(label, num_classes = num_classes)
            batch_labels.append(label)
            batch_data.append(get_data_patch(data, index, patch_size))
    
        yield tf.convert_to_tensor(batch_data), tf.convert_to_tensor(batch_labels)

        batch += 1


def get_valid_indices(data, gt, patch_size, ignored_labels, supervision='full'):
    # Fully supervised : use all pixels with label not ignored
    if supervision == "full":
        mask = np.ones_like(gt)
        for label in ignored_labels:
            mask[gt == label] = 0
    # Semi-supervised : use all pixels, except padding
    elif supervision == "semi":
        mask = np.ones_like(gt)
    x_pos, y_pos = np.nonzero(mask)
    num_neighbors = patch_size // 2
    indices = np.array(
        [
            (x, y)
            for x, y in zip(x_pos, y_pos)
            if x > num_neighbors 
                and x < data.shape[0] - num_neighbors 
                and y > num_neighbors 
                and y < data.shape[1] - num_neighbors
        ]
    )
    # indices = tf.convert_to_tensor(
    #     [
    #         (x, y)
    #         for x, y in zip(x_pos, y_pos)
    #         if x > num_neighbors 
    #             and x < data.shape[0] - num_neighbors 
    #             and y > num_neighbors 
    #             and y < data.shape[1] - num_neighbors
    #     ]
    # )

    labels = np.array([gt[x, y] for x, y in indices])
    # labels = tf.convert_to_tensor([gt[x, y] for x, y in indices])

    return indices, labels

def get_data_patch(data, index, patch_size):
    x, y = index
    x1 = x - patch_size // 2    # Leftmost edge of patch
    y1 = y - patch_size // 2    # Topmost edge of patch
    x2 = x1 + patch_size        # Rightmost edge of patch
    y2 = y1 + patch_size        # Bottommost edge of patch

    patch = data[x1:x2, y1:y2]

    # Copy the data into numpy arrays
    # patch = np.asarray(np.copy(patch), dtype="float32")
    patch = tf.convert_to_tensor(patch, dtype="float32")

    if patch_size == 1:
        patch = patch[:, 0, 0]

    # Add a fourth dimension for 3D CNN
    if patch_size > 1:
        # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        # patch = np.expand_dims(patch, 0)
        patch = tf.expand_dims(patch, 0)
    
    return patch

def get_data_patches(data, indices, patch_size, add_dims=False):
    #TODO

    patches = []

    for index in indices:
        x, y = index
        x1 = x - patch_size // 2    # Leftmost edge of patch
        y1 = y - patch_size // 2    # Topmost edge of patch
        x2 = x1 + patch_size        # Rightmost edge of patch
        y2 = y1 + patch_size        # Bottommost edge of patch

        patch = data[x1:x2, y1:y2]

        # Copy the data into numpy arrays
        # patch = np.asarray(np.copy(patch), dtype="float32")
        patch = tf.convert_to_tensor(patch, dtype="float32")

        if patch_size == 1:
            patch = patch[:, 0, 0]

        # Add a fourth dimension for 3D CNN
        if add_dims and patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            # patch = np.expand_dims(patch, 0)
            patch = tf.expand_dims(patch, 0)

        patches.append(patch)

    # return np.asarray(patches)
    return tf.convert_to_tensor(patches)

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

def load_grss_dfc_2018_uh_dataset(**hyperparams):
    #TODO
    

    dataset = UH_2018_Dataset()
    train_gt = dataset.load_full_gt_image(train_only=True)
    test_gt = dataset.load_full_gt_image(test_only=True)

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
    split_mode = hyperparams['split_mode']

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
    train_gt, val_gt = sample_gt(train_gt, train_split, mode=split_mode)

    train_dataset = HyperspectralDataset(data, train_gt, **hyperparams)
    val_dataset = HyperspectralDataset(data, val_gt, **hyperparams)
    test_dataset = HyperspectralDataset(data, test_gt, shuffle=False, **hyperparams)
    true_test = np.array(test_dataset.labels)

    return train_dataset, val_dataset, test_dataset, true_test

def preprocess_data(data, **hyperparams):
    
    #TODO

    return data

def get_data_split(data, train_gt, test_gt, ignored_labels, 
                   patch_size, validation=True, train_split=0.8, supervision='full'):
    #TODO

    if validation:
        print(f'Splitting training set into {train_split} training, {1-train_split} validation...')
        train_gt, val_gt = sample_gt(train_gt, train_split, mode='fixed')

    print('Getting valid training indices and labels...')
    X_train_indices, y_train = get_valid_indices(data=data, gt=train_gt, 
                                                 patch_size=patch_size, 
                                                 ignored_labels=ignored_labels, 
                                                 supervision=supervision)
    print('Getting training data patches...')
    X_train = get_data_patches(data, X_train_indices, patch_size)
    
    if validation: 
        print('Getting valid validation indices and labels...')
        X_val_indices, y_val = get_valid_indices(data=data, gt=val_gt, 
                                                 patch_size=patch_size, 
                                                 ignored_labels=ignored_labels, 
                                                 supervision=supervision)
        print('Getting validation data patches...')
        X_val = get_data_patches(data, X_val_indices, patch_size)
    
    print('Getting valid testing indices and labels...')
    X_test_indices, y_test = get_valid_indices(data=data, gt=test_gt, 
                                               patch_size=patch_size, 
                                               ignored_labels=ignored_labels, 
                                               supervision=supervision)
    print('Getting testing data patches...')
    X_test = get_data_patches(data, X_test_indices, patch_size)

    print('Data split completed!')
    if validation:
        return (X_train, y_train, X_test, y_test, X_val, y_val)
    else:
        return (X_train, y_train, X_test, y_test)