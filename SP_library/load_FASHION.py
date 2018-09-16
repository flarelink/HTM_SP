import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn as skl
import cv2


def load_FASHION(train_csv, test_csv):
    """
    Takes in Fashion-MNIST csv files to create numpy train/test data and label arrays using pandas.

    :param train_csv: name of Fashion-MNIST train csv, i.e.) FMN_Im32x32_tr.csv
    :param test_csv: name of Fashion-MNIST test csv, i.e.) FMN_Im32x32_ts.csv
    :return: train data array
             train labels array
             test data array
             test labels array
    """

    # Get current working directory
    cwd = os.getcwd()

    # Load train data and train labels
    train_data = pd.read_csv(cwd + '/SP_library/FASHION/' + train_csv, header=None)
    train_data = train_data.drop(train_data.columns[0], axis=1).values
    train_labels = pd.read_csv(cwd + '/SP_library/FASHION/' + train_csv, header=None, usecols=[0]).values

    # Load test data and test labels
    test_data = pd.read_csv(cwd + '/SP_library/FASHION/' + test_csv, header=None)
    test_data = test_data.drop(test_data.columns[0], axis=1).values
    test_labels = pd.read_csv(cwd + '/SP_library/FASHION/' + test_csv, header=None, usecols=[0]).values

    return train_data, train_labels, test_data, test_labels