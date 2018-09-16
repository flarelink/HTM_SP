import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn as skl
import cv2
from load_MNIST import load_MNIST

def bin_MNIST(train_csv, test_csv, pixel_size):
    """
    Loads MNIST files and binarizes them to a resized value such as 16x16 or 32x32
    which is then outputted as a new csv file.

    :param train_csv: Location of original MNIST csv train data
    :param test_csv: Location of original MNIST csv test data
    :param pixel_size: Specified number of pixels, i.e.) 16x16 or 32x32
    :return: No returns, but csv files will be created
    """

    # Get current working directory
    cwd = os.getcwd()

    # Load MNIST data
    train_data, train_labels, test_data, test_labels = load_MNIST(train_csv, test_csv)

    # Initialize binary arrays for MNIST
    bin_train_data = np.zeros((60000, ((pixel_size * pixel_size))), dtype=int)
    bin_test_data = np.zeros((10000, ((pixel_size * pixel_size))), dtype=int)

    # Reshape MNIST to pixel_size x pixel_size
    for i in xrange(len(train_data)):
        data = train_data[i, :]

        plottable_image = np.reshape(data, (28, 28)).astype(np.uint8)
        resized_image = cv2.resize(plottable_image, (pixel_size, pixel_size))
        bin_train_data[i, :] = resized_image.flatten()

    for j in xrange(len(test_data)):
        data = test_data[j, :]

        plottable_image = np.reshape(data, (28, 28)).astype(np.uint8)
        resized_image = cv2.resize(plottable_image, (pixel_size, pixel_size))
        bin_test_data[j, :] = resized_image.flatten()

    # Binarize images
    bin_train_data[bin_train_data > 0] = 1
    bin_test_data[bin_test_data > 0] = 1

    # Add labels to binary data
    # print(bin_train_data.shape)
    # print(train_labels.shape)
    bin_train_data = np.insert(bin_train_data, [0], train_labels, axis=1)
    bin_test_data = np.insert(bin_test_data, [0], test_labels, axis=1)

    # Output CSV files
    cols = pixel_size * pixel_size
    np.savetxt(cwd + "/SP_library/MNIST/MNIST_train_bin_{}.csv".format(cols), bin_train_data, delimiter=',')
    np.savetxt(cwd + "/SP_library/MNIST/MNIST_test_bin_{}.csv".format(cols), bin_test_data, delimiter=',')

    # Visualize MNIST Images
    # for i in xrange(3):
    #     image = bin_train_data[i, :]
    #     label = train_labels[i, 0]
    #
    #     plottable_image = np.reshape(image, (pixel_size, pixel_size))
    #
    #     # Plot the image
    #     plt.imshow(plottable_image, cmap='gray_r')
    #     plt.title('Train Digit Label: {}'.format(label))
    #     plt.show()

    #     image = test_data[i, :]
    #     label = test_labels[i, 0]
    #
    #     plottable_image = np.reshape(image, (28, 28))
    #
    #     # Plot the image
    #     plt.imshow(plottable_image, cmap='gray_r')
    #     plt.title('Test Digit Label: {}'.format(label))
    #     plt.show()

    train_bin_csv = 'MNIST_train_bin_{}.csv'.format(cols)
    test_bin_csv = 'MNIST_test_bin_{}.csv'.format(cols)

    return train_bin_csv, test_bin_csv