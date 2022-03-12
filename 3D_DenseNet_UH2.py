#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test harness module for 3D-DenseNet for University of Houston 2018
"""

### Built-in Imports ###
import argparse
from cgi import test
import collections
import math
import os
import time

### Other Library Imports ###
import numpy as np
from numpy.core.numeric import full_like
import scipy.io as sio
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.callbacks as kcallbacks
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical, Sequence

### Local Imports ###
from grss_dfc_2018_uh import NUMBER_OF_UH_2018_CLASSES, UH_2018_Dataset
from Utils import averageAccuracy, densenet_IN, modelStatsRecord, zeroPadding

### Environment Setup ###
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### Global Constants ###

class Dataset(Sequence):
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
        super(Dataset, self).__init__()
        self.data = data
        self.label = gt
        self.batch_size = hyperparams["batch_size"]
        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.flip_augmentation = hyperparams["flip_augmentation"]
        self.radiation_augmentation = hyperparams["radiation_augmentation"]
        self.mixture_augmentation = hyperparams["mixture_augmentation"]
        self.center_pixel = hyperparams["center_pixel"]
        self.n_classes = hyperparams['n_classes']
        self.one_hot_encoding = hyperparams['one_hot_encoding']
        supervision = hyperparams["supervision"]
        # Fully supervised : use all pixels with label not ignored
        if supervision == "full":
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == "semi":
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]
        if shuffle:
            np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1.0, size=2)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert self.labels[l_indice] == value
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

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

            if self.flip_augmentation and self.patch_size > 1:
                # Perform data augmentation (only on 2D patches)
                data, label = self.flip(data, label)
            if self.radiation_augmentation and np.random.random() < 0.1:
                data = self.radiation_noise(data)
            if self.mixture_augmentation and np.random.random() < 0.2:
                data = self.mixture_noise(data, label)

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

            if self.one_hot_encoding:
                label = to_categorical(label, num_classes = self.n_classes)

            data_batch.append(data)
            labels.append(label)

        # data_batch = tf.convert_to_tensor(data_batch)
        # labels = tf.convert_to_tensor(labels)
        data_batch = np.asarray(data_batch)
        labels = np.asarray(labels)

        return data_batch, labels

def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = tf.device('/CPU:0')
    elif tf.test.is_gpu_available(cuda_only=True):
        print(f'Computation on CUDA GPU device {ordinal}')
        device = tf.device(f'/GPU:{ordinal}')
        # tf.config.experimental.set_memory_growth(device, True)
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = tf.device('/CPU:0')
    return device

def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)

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
       print("Sampling {} with train size = {}".format(mode, train_size))
       train_indices, test_indices = [], []
       for c in np.unique(gt):
           if c == 0:
              continue
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features

           train, test = train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
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
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt



def prime_generator():
    """ 
    Generate an infinite sequence of prime numbers.

    Sieve of Eratosthenes
    Code by David Eppstein, UC Irvine, 28 Feb 2002
    http://code.activestate.com/recipes/117119/
    """
    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}
    
    # The running integer that's checked for primeness
    q = 2
    
    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            # 
            yield q
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next 
            # multiples of its witnesses to prepare for larger
            # numbers
            # 
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]
        
        q += 1

def model_DenseNet(img_rows, img_cols, img_channels, nb_classes, lr=0.001, momentum=0.0):
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
    lr : float
        Learning rate for the model

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
    RMS = RMSprop(learning_rate=lr, momentum=momentum)

    # Compile DenseNet model
    # model_dense.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])
    model_dense.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])

    return model_dense

def run_3d_densenet_uh(**hyperparams):
    """
    Runs the 3D-DenseNet for the University of Houston dataset.
    """

    dataset = UH_2018_Dataset()
    train_gt = dataset.load_full_gt_image(train_only=True)
    # train_gt = to_categorical(train_gt).reshape(train_gt.shape + (dataset.gt_num_classes + 1,))
    test_gt = dataset.load_full_gt_image(test_only=True)
    # test_gt = to_categorical(test_gt).reshape(test_gt.shape + (dataset.gt_num_classes + 1,))

    # hs_data = dataset.load_full_hs_image()
    # lidar_ms_data = dataset.load_full_lidar_ms_image()
    # data = np.dstack((hs_data, lidar_ms_data))
    data = None

    if hyperparams['use_hs_data'] or hyperparams['use_all_data']:
        hs_data = dataset.load_full_hs_image()
        print(f'hs_data shape: {hs_data.shape}')
        if data is None:
            data = np.copy(hs_data)
        else:
            data = np.dstack((data, hs_data))

    if hyperparams['use_lidar_ms_data'] or hyperparams['use_all_data']:
        lidar_ms_data = dataset.load_full_lidar_ms_image()
        print(f'lidar_ms_data shape: {lidar_ms_data.shape}')
        if data is None:
            data = np.copy(lidar_ms_data)
        else:
            data = np.dstack((data, lidar_ms_data))

    if hyperparams['use_lidar_ndsm_data'] or hyperparams['use_all_data']:
        lidar_ndsm_data = dataset.load_full_lidar_ndsm_image()
        print(f'lidar_ndsm_data shape: {lidar_ndsm_data.shape}')
        if data is None:
            data = np.copy(lidar_ndsm_data)
        else:
            data = np.dstack((data, lidar_ndsm_data))

    if hyperparams['use_vhr_data'] or hyperparams['use_all_data']:
        vhr_data = dataset.load_full_vhr_image()
        print(f'vhr_data shape: {vhr_data.shape}')
        if not data:
            data = np.copy(vhr_data)
        else:
            data = np.dstack((data, vhr_data))
    
    if data is not None:
        print(f'data shape: {data.shape}')
    else:
        print('No data was loaded! Training cancelled...')
        return

    print(f'train_gt shape: {train_gt.shape}')
    print(f'test_gt shape: {test_gt.shape}')

    hyperparams.update(
        {
            "n_classes": dataset.gt_num_classes,
            "n_bands": data.shape[-1],
            "ignored_labels": dataset.gt_ignored_labels,
            "device": get_device(args.cuda),
            "supervision": "full",
            "center_pixel": True,
            "one_hot_encoding": True,
        }
    )
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)    

    ### Set Variables ###

    iterations = hyperparams['iterations']  # num iterations to run model
    patience = hyperparams['patience']      # num epochs w/o improvement before stopping training
    epochs = hyperparams['epochs']          # number of epochs per iteration
    batch_size = ['batch_size']             # number of samples per batch
    patch_size = hyperparams['patch_size']  # N in NxN patch per sample
    train_split = hyperparams['train_split']    # training percent in val/train split
    classes = dataset.gt_num_classes        # number of label categories
    bands = data.shape[-1]  # number of spectral bands
    img_rows = patch_size   # number of rows in neighborhood
    img_cols = patch_size   # number of cols in neighborhood
    lr = hyperparams['lr']  # learning rate for model
    momentum = hyperparams['momentum']  # optimizer momentum
    
    pad = patch_size // 2

    # Pad only first two dimensions
    data = np.pad(data, [(pad,), (pad,), (0,)], mode='constant')
    train_gt = np.pad(train_gt, [(pad,), (pad,)], mode='constant')
    test_gt = np.pad(test_gt, [(pad,), (pad,)], mode='constant')

    print(f'padded data shape: {data.shape}')
    print(f'padded train_gt shape: {train_gt.shape}')
    print(f'padded test_gt shape: {test_gt.shape}')

    train_gt, val_gt = sample_gt(train_gt, train_split, mode='random')

    print(f'train_gt shape after sample: {train_gt.shape}')
    print(f'val_gt shape: {val_gt.shape}')

    train_dataset = Dataset(data, train_gt, **hyperparams)
    val_dataset = Dataset(data, val_gt, **hyperparams)
    test_dataset = Dataset(data, test_gt, shuffle=False, **hyperparams)

    # Delete unused memory
    # dataset.clear_all_images()

    # Initialize statistics lists
    KAPPA_3D_DenseNet = []
    OA_3D_DenseNet = []
    AA_3D_DenseNet = []
    TRAINING_TIME_3D_DenseNet = []
    TESTING_TIME_3D_DenseNet = []
    ELEMENT_ACC_3D_DenseNet = np.zeros((iterations, classes))

    # Run 3-D DenseNet for ITER iterations
    for index_iter in range(iterations):
        print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
        print(f'>>> Iteration #{index_iter + 1} >>>')

        # Path for saving the best validated model at the model
        # checkpoint
        best_weights_DenseNet_path = 'training_results/university_of_houston/UHouston_best_3D_DenseNet_1' + str(
            index_iter + 1) + '.hdf5'

        # Initialize random seed for sampling function
        # Each random seed is a prime number, in order
        seed = next(prime_generator())
        print(f'Iteration #{index_iter} seed: {seed}')
        np.random.seed(seed)

        ############################################################################################################
        # Model creation, training, and testing
        print(f'img_rows: {img_rows}')
        print(f'img_cols: {img_cols}')
        print(f'bands: {bands}')
        print(f'classes: {classes}')
        print(f'learning rate: {lr}')
        print(f'momentum: {momentum}')

        model_densenet = model_DenseNet(img_rows, img_cols, bands, classes, 
                                        lr=lr, momentum=momentum)

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
            train_dataset,
            validation_data=val_dataset,
            batch_size=batch_size,
            epochs=epochs, 
            shuffle=True, 
            callbacks=[cb_early_stopping, cb_save_best_model])
        
        # Record end time for model training
        model_train_end = time.process_time()

        # Record start time for model evaluation
        model_test_start = time.process_time()

        # Evaluate the trained 3D-DenseNet
        loss_and_metrics = model_densenet.evaluate(
            test_dataset,
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
        # pred_test = model_densenet.predict(test_dataset).argmax(axis=1)
        pred_test = model_densenet.predict(test_dataset).argmax(axis=1)
        target_test = np.array(test_dataset.labels)
        
        print(f'target_test shape: {target_test.shape}')
        print(f'pred_test shape: {pred_test.shape}')

        # mask = np.ones_like(test_gt)
        # for l in dataset.gt_ignored_labels:
        #     mask[test_gt == l] = 0

        # test_gt = test_gt[np.nonzero(mask)].ravel()

        # Store the prediction label counts
        collections.Counter(pred_test)
        
        # print(f'test_gt shape: {test_gt.shape}')
        # print(f'pred_test shape: {pred_test.shape}')

        # Get prediction accuracy metric
        overall_acc = metrics.accuracy_score(target_test, pred_test)
        
        # Get prediction confusion matrix
        confusion_matrix = metrics.confusion_matrix(target_test, pred_test)
        
        # Get individual class accuracy as well as average accuracy
        each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
        
        # Get Kappa metric from predictions
        kappa = metrics.cohen_kappa_score(target_test, pred_test)
        
        # Append all metrics to their respective lists
        KAPPA_3D_DenseNet.append(kappa)
        OA_3D_DenseNet.append(overall_acc)
        AA_3D_DenseNet.append(average_acc)

        # Append training and testing times to their respective lists
        TRAINING_TIME_3D_DenseNet.append(model_train_end - model_train_start)
        TESTING_TIME_3D_DenseNet.append(model_test_end - model_test_start)
        
        # Save individual accuracies to iteration index in element
        # accuracy list
        ELEMENT_ACC_3D_DenseNet[index_iter, :] = np.insert(each_acc, 0 ,0)

        print("3D DenseNet finished.")
        print(f'<<< Iteration #{index_iter + 1} <<<')
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    # Print out the overall training and testing results for the model
    # and save the results to a file
    modelStatsRecord.outputStats(KAPPA_3D_DenseNet, OA_3D_DenseNet, AA_3D_DenseNet, ELEMENT_ACC_3D_DenseNet,
                                TRAINING_TIME_3D_DenseNet, TESTING_TIME_3D_DenseNet,
                                history_3d_densenet, loss_and_metrics, classes,
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
    parser.add_argument(
        "--cuda",
        type=int,
        default=-1,
        help="Specify CUDA device (defaults to -1, which learns on CPU)",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
    parser.add_argument(
        "--restore",
        type=str,
        default=None,
        help="Weights to use for initialization, e.g. a checkpoint",
    )

    # Training options
    group_train = parser.add_argument_group("Training")
    group_train.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs (default = 1)",
    )
    group_train.add_argument(
        "--patch_size",
        type=int,
        default=3,
        help="Size of the spatial neighborhood (default = 3)",
    )
    group_train.add_argument(
        "--lr", 
        type=float, 
        default = 0.001,
        help="Learning rate (default = 0.001)"
    )
    group_train.add_argument(
        "--momentum", 
        type=float, 
        default = 0.0,
        help="Momentum (default = 0.0)"
    )
    group_train.add_argument(
        "--train_split", 
        type=float, 
        default = 0.80,
        help="The amount of samples set aside for training \
              during validation split (default = 0.80)"
    )
    group_train.add_argument(
        "--class_balancing",
        action="store_true",
        help="Inverse median frequency class balancing (default = False)",
    )
    group_train.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default = 64)",
    )
    group_train.add_argument(
        "--test_stride",
        type=int,
        default=1,
        help="Sliding window step stride during inference (default = 1)",
    )
    group_train.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the model for (default = 1)",
    )
    group_train.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs without improvement before stopping training",
    )
    # Data augmentation parameters
    group_da = parser.add_argument_group("Data augmentation")
    group_da.add_argument(
        "--flip_augmentation", action="store_true", help="Random flips (if patch_size > 1)"
    )
    group_da.add_argument(
        "--radiation_augmentation",
        action="store_true",
        help="Random radiation noise (illumination)",
    )
    group_da.add_argument(
        "--mixture_augmentation", action="store_true", help="Random mixes between spectra"
    )

    # Dataset parameters
    group_ds = parser.add_argument_group("Dataset")
    group_ds.add_argument(
        "--use_hs_data", action="store_false", help="Use hyperspectral data"
    )
    group_ds.add_argument(
        "--use_lidar_ms_data", action="store_true", help="Use lidar multispectral intensity data"
    )
    group_ds.add_argument(
        "--use_lidar_ndsm_data", action="store_true", help="Use lidar NDSM data"
    )
    group_ds.add_argument(
        "--use_vhr_data", action="store_true", help="Use very high resolution RGB data"
    )
    group_ds.add_argument(
        "--use_all_data", action="store_true", help="Use all data sources"
    )

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

    hyperparams = vars(args)

    # Run Model
    run_3d_densenet_uh(**hyperparams)
