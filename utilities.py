#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File with utility functions and variables.
"""

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