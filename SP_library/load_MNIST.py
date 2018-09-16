import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn as skl
import cv2


def load_MNIST(train_csv, test_csv):
    """
    Takes in MNIST csv files to create numpy train/test data and label arrays using pandas.

    :param train_csv: name of MNIST train csv, i.e.) MNIST_train.csv
    :param test_csv: name of MNIST test csv, i.e.) MNIST_test.csv
    :return: train data array
             train labels array
             test data array
             test labels array
    """

    # Get current working directory
    cwd = os.getcwd()

    # Load train data and train labels
    train_data = pd.read_csv(cwd + '/SP_library/MNIST/' + train_csv, header=None)
    train_data = train_data.drop(train_data.columns[0], axis=1).values
    train_labels = pd.read_csv(cwd + '/SP_library/MNIST/' + train_csv, header=None, usecols=[0]).values

    # Load test data and test labels
    test_data = pd.read_csv(cwd + '/SP_library/MNIST/' + test_csv, header=None)
    test_data = test_data.drop(test_data.columns[0], axis=1).values
    test_labels = pd.read_csv(cwd + '/SP_library/MNIST/' + test_csv, header=None, usecols=[0]).values

    return train_data, train_labels, test_data, test_labels