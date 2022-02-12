#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test harness module for Automatic Hyperspectral Band Selection

This script is the test harness for experimenting on automatic
hyperspectral (HS) band selection for classifying hyperspectral images.

Author:  Christopher Good
Version: 1.0.0

Usage: test_harness.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###
import argparse

### Other Library Imports ###
#TODO

### Local Imports ###
#TODO

### Constants ###
SCRIPT_DESCRIPTION = ('Test harness script for experimenting on automatic '
                      'hyperspectral (HS) band selection for the classification'
                      'of HS images.')

### Definitions ###

def test_harness_parser():
    """
    Sets up the parser for command-line flags for the test harness 
    script.

    Returns
    -------
    argparse.ArgumentParser
        An ArgumentParser object configured with the test_harness.py
        command-line arguments.
    """

    parser = argparse.ArgumentParser(SCRIPT_DESCRIPTION)
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
    parser = test_harness_parser()
    args = parser.parse_args()

    dataset_path = args.dataset_path
    outfile_prefix = args.test_name
    batch_size = args.batch_size
    num_classes = args.number_of_classes
    epochs = args.epochs
    validation_split = args.validation_split
    image_rows = args.image_rows
    image_cols = args.image_cols
    verbose = args.verbose
