import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn as skl
import cv2
import math

def sparseness(n_cols, sdr):
    """
    Sparsity metric to check if formed sparsity is fixed
    :param n_cols: number of columns
    :param sdr: sdr input
    :return: s: sparsity metric
    """
    s = (1.0/float(n_cols)) * np.sum(sdr)
    return s

def entropy(sdr_train_array, sdr_test_array):
    """
    Entropy metric
    :param sdr_train_array: sdrs formed from training data
    :param sdr_test_array: sdrs formed from testing data
    :return: train_entropy: entropy for training data
                 test_entropy: entropy for testing data
    """
    # train_P and test_P = P(a_i)
    # entropy = S
    n_cols = len(sdr_train_array[0])
    train_P = np.zeros([1, n_cols])
    test_P = np.zeros([1, n_cols])
    train_entropy_array = np.zeros([1, n_cols])
    test_entropy_array = np.zeros([1, n_cols])

    for c in xrange(n_cols):
        # calculating training entropy
        train_P[0, c] = (1.0/float(len(sdr_train_array))) * np.sum(sdr_train_array[0, c], dtype=float)
        print((1.0/float(len(sdr_train_array))))
        print(np.sum(sdr_train_array[0, c], dtype=float))
        print(train_P[0, c])
        train_entropy_array[0, c] = -train_P[0, c] * math.log(train_P[0, c], 2.0) - \
                              (1 - train_P[0, c]) * math.log((1 - train_P[0, c]), 2.0)

        # calculating testing entropy
        test_P[0, c] = (1.0 / float(len(sdr_test_array))) * np.sum(sdr_test_array[0, c])
        test_entropy_array[0, c] = -test_P[0, c] * math.log(test_P[0, c], 2.0) - \
                              (1 - test_P[0, c]) * math.log((1 - test_P[0, c]), 2.0)

    # Final entropy calculation
    train_entropy = np.sum(train_entropy_array)
    test_entropy = np.sum(test_entropy_array)

    return train_entropy, test_entropy
