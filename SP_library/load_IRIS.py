"""
Author: Humza Syed
Description: Loads Iris dataset from a csv and outputs train/test data and label arrays
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_IRIS(iris_csv):
    """
    Takes in Iris csv files to create numpy train/test data and label arrays using pandas.

    :param iris_csv: name of csv file; example) iris.csv
    :return: train data array
             train labels array
             test data array
             test labels array
    """

    # Get current working directory
    cwd = os.getcwd()

    # Load data and labels
    input_data = pd.read_csv(cwd + '/SP_library/IRIS/' + iris_csv, header=None)
    input_data = input_data.drop(input_data.columns[4], axis=1).values

    input_labels = pd.read_csv(cwd + '/SP_library/IRIS/' + iris_csv, header=None, usecols=[4]).values.flatten()
    lu_unique = np.unique(input_labels)
    lu_map = {k:i for (i,k) in enumerate(sorted(lu_unique))}
    input_labels = [lu_map[k] for k in input_labels]
    input_labels = np.array(input_labels)

    # Split into train/test data and train/test labels
    train_data, test_data, train_labels, test_labels = train_test_split(input_data, input_labels,
                                                                        train_size=0.8, random_state=42)

    # np.set_printoptions(threshold=np.inf)

    return train_data, train_labels, test_data, test_labels