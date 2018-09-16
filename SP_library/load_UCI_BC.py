"""
Author: Humza Syed
Description: Loads UCI Breast Cancer dataset from a csv and outputs train/test data and label arrays
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_UCI_BC(uci_bc_csv):
    """
    Takes in UCI breast cancer csv files to create numpy train/test data and label arrays using pandas.

    DISCLAIMER: CONVERTING ALL '?'s IN UCI BC TO VALUES OF 1

    :param uci_bc_csv: name of csv file; example) breast_cancer.csv
    :return: train data array
             train labels array
             test data array
             test labels array
    """

    # Get current working directory
    cwd = os.getcwd()

    # Load data and labels
    input_data = pd.read_csv(cwd + '/SP_library/UCI_BC/' + uci_bc_csv, header=None, na_values='?') # set all ?'s to NaN
    input_data = input_data.fillna(value=1)  # convert all NaN --> 1
    input_data = input_data.drop(input_data.columns[0], axis=1).values
    input_data = np.delete(input_data, 9, 1)

    input_labels = pd.read_csv(cwd + '/SP_library/UCI_BC/' + uci_bc_csv, header=None, usecols=[10]).values.flatten()
    lu_unique = np.unique(input_labels)
    lu_map = {k:i for (i,k) in enumerate(sorted(lu_unique))}
    input_labels = [lu_map[k] for k in input_labels]
    input_labels = np.array(input_labels)

    """
    Thank you Zach
    >>> xs_uniq = np.unique(xs)
    >>> xs_map = {k: i for (i, k) in enumerate(sorted(xs_uniq))}
    >>> x = [xs_map[k] for k in xs]
    """

    # Split into train/test data and train/test labels
    train_data, test_data, train_labels, test_labels = train_test_split(input_data, input_labels,
                                                                        train_size=0.8, random_state=42)

    # np.set_printoptions(threshold=np.inf)

    return train_data, train_labels, test_data, test_labels