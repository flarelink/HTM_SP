import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from load_MNIST import load_MNIST
import scipy.sparse


def softmax_classify(sdr_train_array, sdr_train_labels, sdr_test_array, sdr_test_labels, n_classes, n_epochs, lr):
    """
    Softmax classifier to determine if algorithm classified correctly
    :param sdr_train_array  : sdrs formed from training data
    :param sdr_train_labels : labels for training data
    :param sdr_test_array   : sdrs formed from testing data
    :param sdr_test_labels  : labels for testing data
    :param n_classes        : number of classes
    :param n_epochs         : number of training epochs
    :param lr               : learning rate for classifier
    :return: final_train_acc: final calculated training accuracy
             final_test_acc : final calculated testing acccuracy
    """
    # Initialize some helpful values
    n_cols = len(sdr_train_array[0])
    n_train_ex = len(sdr_train_array)
    n_test_ex =  len(sdr_test_array)

    # matrices for plotting data
    train_accuracy_iter = np.zeros((1, n_train_ex), dtype=float)
    test_accuracy_iter = np.zeros((1, n_test_ex), dtype=float)
    train_accuracy_array = np.zeros(n_epochs, dtype=float)
    test_accuracy_array = np.zeros(n_epochs, dtype=float)

    # weight matrix of size n_cols x n_classes
    w = np.random.uniform(low=0.0, high=1.0, size=(n_cols, n_classes))

    # one hot encode labels
    # np.set_printoptions(threshold=np.inf)

    one_hot_train_labels = (np.arange(np.max(sdr_train_labels) + 1) == sdr_train_labels[:, None]).astype(float)
    # print('one-hot encoding:\n', one_hot_train_labels)
    # one_hot_test_labels = (np.arange(np.max(sdr_test_labels) + 1) == sdr_test_labels[:, None]).astype(float)
    # print('one-hot encoding:\n', one_hot_test_labels)

    for e in range(n_epochs):
        for i in range(n_train_ex):
            s = np.reshape(sdr_train_array[i, :], (1, n_cols))
            logit = np.dot(s, w)
            softmax = np.exp(logit) / np.sum(np.exp(logit))
            # train weights
            soft_label = (softmax - one_hot_train_labels[i])
            dw = np.dot(np.transpose(s), (lr * soft_label))
            w = w - dw

            train_accuracy_iter[0, i] = np.argmax(softmax) == sdr_train_labels[i]
        train_accuracy = np.mean(train_accuracy_iter)

        for i in range(n_test_ex):
            s = np.reshape(sdr_test_array[i, :], (1, n_cols))
            logit = np.dot(s, w)
            softmax = np.exp(logit) / np.sum(np.exp(logit))
            test_accuracy_iter[0, i] = np.argmax(softmax) == sdr_test_labels[i]
        test_accuracy = np.mean(test_accuracy_iter)

        if(e % 10 == 0 or e == (n_epochs - 1) ):
            print(' Current epoch is: %d' % e)
            print('Training Accuracy: %f' % train_accuracy)
            print('    Test Accuracy: %f' % test_accuracy)
        train_accuracy_array[e] = train_accuracy
        test_accuracy_array[e] = test_accuracy

        if(e == (n_epochs - 1)):
            final_train_acc = train_accuracy
            final_test_acc = test_accuracy

    plt.figure(1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(train_accuracy_array, 'r', label='training')
    plt.plot(test_accuracy_array, 'b', label='testing')
    plt.legend(loc='lower right')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    return final_train_acc, final_test_acc
