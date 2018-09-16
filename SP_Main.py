# =============================================================================
# SP_Main.py - Spatial Pooler Implementation of Hierarchical Temporal Memory
# Copyright (C) 2018  Humza Syed
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn as skl
import cv2
import argparse
import sys
sys.path.insert(0, './SP_library')

from SP_init import SP_init
from classifiers import softmax_classify
from metrics import entropy
from load_MNIST import load_MNIST
from bin_MNIST import bin_MNIST
from load_UCI_BC import load_UCI_BC
from load_IRIS import load_IRIS
from load_FASHION import load_FASHION
from encoders import scalar_encoder

if __name__ == "__main__":

    # Basic function to allow for parsing of true/false
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Initialize parser
    parser = argparse.ArgumentParser(description='Spatial Pooler of Hierarchical Temporal Memory Algorithm')

    # Binarized MNIST dataset
    parser.add_argument('--MNIST', type=str2bool, nargs='?', default=True,
                        help='Uses the binarized MNIST dataset if set to True; (default=True')
    parser.add_argument('--bin_MNIST', type=int, default=28,
                        help='Create a new binarized csv file of MNIST by specifying pixel size, i.e.) 16x16, pixel size = 16; (default=16)')
    parser.add_argument('--MNIST_tr_csv', type=str, default='MNIST_train_bin_256.csv',
                        help='Input binarized train csv file for MNIST, use binary forms; (default=MNIST_train_bin_256.csv)')
    parser.add_argument('--MNIST_ts_csv', type=str, default='MNIST_test_bin_256.csv',
                        help='Input binarized test csv file for MNIST, use binary forms; (default=MNIST_test_bin_256.csv)')

    # Breast cancer dataset
    parser.add_argument('--uci_bc', type=str2bool, nargs='?', default=False,
                        help='Uses the UCI breast cancer dataset if set to True; (default=False')
    parser.add_argument('--uci_bc_csv', type=str, default='breast_cancer_orig.csv',
                        help='Input csv file for UCI breast cancer; (default=breast_cancer.csv)')

    # Iris dataset
    parser.add_argument('--iris', type=str2bool, nargs='?', default=False,
                        help='Uses the Iris dataset if set to True; (default=False')
    parser.add_argument('--iris_csv', type=str, default='iris.csv',
                        help='Input csv file for iris dataset; (default=iris.csv)')

    # Binarized FASHION-MNIST dataset
    parser.add_argument('--FASHION', type=str2bool, nargs='?', default=False,
                        help='Uses the binarized MNIST dataset if set to True; (default=True')
    parser.add_argument('--FM_tr_csv', type=str, default='FMN_Im32x32_tr.csv',
                        help='Input binarized train csv file for FASHION-MNIST, use binary forms; (default=FMN_Im32x32_tr.csv)')
    parser.add_argument('--FM_ts_csv', type=str, default='FMN_Im32x32_ts.csv',
                        help='Input binarized test csv file for FASHION-MNIST, use binary forms; (default=FMN_Im32x32_ts.csv)')

    # Encoder arguments
    parser.add_argument('--enc_n', type=int, default=1024,
                        help='Creates an output representation of size n for each input; (default=1024)')
    parser.add_argument('--enc_w', type=int, default=39,
                        help='Determines number of active bits for output, should be changed from default and should be an odd value; (default=39)')
    parser.add_argument('--enc_periodic', type=str2bool, nargs='?', default=False,
                        help='Flag to say if input data is periodic; (default=False')
    parser.add_argument('--enc_clip_input', type=str2bool, nargs='?', default=False,
                        help='Flag to say if input data should be clipped; (default=False')

    # Spatial Pooler initializations
    parser.add_argument('--n_cols', type=int, default=256,
                        help='Number of columns in SP; (default=256)')
    parser.add_argument('--n_proxim_con', type=int, default=31,
                        help='Number of proximal connections between cols; (default=31)')
    parser.add_argument('--perm_thresh', type=float, default=0.5,
                        help='Permanence threshold; (default=0.5)')
    parser.add_argument('--perm_inc', type=float, default=0.01,
                        help='Amount that permanence value increases from winner-take-all; (default=0.01)')
    parser.add_argument('--perm_dec', type=float, default=-0.01,
                        help='Amount that permanence value decreases from winner-take-all, it is -0.02 because forgetting works better; (default=-0.01)')
    parser.add_argument('--min_overlap', type=int, default=2,
                        help='Minimum overlap score needed for winner-take-all; (default=2)')
    parser.add_argument('--n_winners', type=int, default=40,
                        help='Number of winners for winner-take-all; (default=40)')
    parser.add_argument('--beta_boost', type=float, default=3,
                        help='Factor by which overlap scores are boosted; (default=3)')
    parser.add_argument('--T_boost_speed', type=int, default=1000,
                        help='Controls how fast the boost factors are updated; (default=1000)')

    # Softmax hyperparameters
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes in labels; (default=10)')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of epochs to train the softmax classifier; (default=50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for softmax classifier; (default=0.001)')

    # Data parameters
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of runs for the program')
    parser.add_argument('--verbose', type=str2bool, nargs='?', default=False,
                        help='Verbosity flag to show image of data and sdr if set to True; (default=False)')

    args = parser.parse_args()

    # Defaults for checking
    train_data = 0
    test_data = 0
    train_labels = 0
    test_labels = 0
    train_acc_array = np.zeros(args.n_runs, dtype=float)
    test_acc_array  = np.zeros(args.n_runs, dtype=float)
    cwd = os.getcwd()

    ##### RUN FOR SET AMOUNT OF TIMES #####
    for r in xrange(args.n_runs):

         ##### LOAD DATASET, HIERARCHY SYSTEM WHERE MNIST WILL BE CHOSEN FIRST IF TWO DATASETS ARE SET TO TRUE #####
    
        # Load MNIST data
        if(args.MNIST == True):
            # check if pixel binarized file is already made
            if ( (args.bin_MNIST != 28) or (os.path.exists(cwd + '/SP_library/MNIST/MNIST_train_bin_{}.csv'.format(args.bin_MNIST**2) ) == False) ):
                args.MNIST_tr_csv, args.MNIST_ts_csv = bin_MNIST('mnist_train_orig.csv', 'mnist_test_orig.csv', args.bin_MNIST)

            # Load data
            print('MNIST dataset being loaded')
            train_data, train_labels, test_data, test_labels = load_MNIST(args.MNIST_tr_csv, args.MNIST_ts_csv)
            print('MNIST loading completed')

        # Load UCI Breast Cancer dataset
        elif(args.uci_bc == True):
            # Load data
            print('UCI Breast Cancer dataset being loaded')
            pre_train_data, train_labels, pre_test_data, test_labels = load_UCI_BC(args.uci_bc_csv)
            print('UCI Breast Cancer loading completed')

            # Scalar Encoding for HTM
            print('Encoding process beginning')
            train_data = scalar_encoder(pre_train_data, args.enc_n, args.enc_w, args.enc_periodic, args.enc_clip_input)
            test_data =  scalar_encoder(pre_test_data,  args.enc_n, args.enc_w, args.enc_periodic, args.enc_clip_input)
            print('Encoding completed')

        # Load Iris dataset
        elif(args.iris == True):
            print('Iris dataset being loaded')
            pre_train_data, train_labels, pre_test_data, test_labels = load_IRIS(args.iris_csv)
            print('Iris loading completed')

            # Scalar Encoding for HTM
            print('Encoding process beginning')
            train_data = scalar_encoder(pre_train_data, args.enc_n, args.enc_w, args.enc_periodic, args.enc_clip_input)
            test_data = scalar_encoder(pre_test_data, args.enc_n, args.enc_w, args.enc_periodic, args.enc_clip_input)
            print('Encoding completed')

        # Load FASHION-MNIST dataset
        elif(args.FASHION == True):
            # Load data
            print('FASHION-MNIST dataset being loaded')
            train_data, train_labels, test_data, test_labels = load_FASHION(args.FM_tr_csv, args.FM_ts_csv)
            print('FASHION-MNIST loading completed')
            
        # Basic check to see if any dataset was even selected
        else:
            raise IOError('No dataset is selected')

        ##### RUNNING DATA THROUGH SPATIAL POOLER #####
        print('Spatial pooler commencing...')
        sdr_train_array, sdr_test_array, \
        sparse_train_array, sparse_test_array              = SP_init(train_data,
                                                                     test_data,
                                                                     args.n_cols,
                                                                     args.n_proxim_con,
                                                                     args.perm_thresh,
                                                                     args.perm_inc,
                                                                     args.perm_dec,
                                                                     args.min_overlap,
                                                                     args.n_winners,
                                                                     args.beta_boost,
                                                                     args.T_boost_speed,
                                                                     args.verbose)
        print('Spatial pooling completed')

        ##### TRAIN SOFTMAX CLASSIFIER #####
        print('Softmax classifier commencing...')
        train_acc, test_acc = softmax_classify(sdr_train_array, train_labels,
                                               sdr_test_array, test_labels,
                                               args.n_classes, args.n_epochs, args.lr)
        train_acc_array[r] = train_acc
        test_acc_array[r]  = test_acc
        print('Softmax classification completed')

        # Print out metrics
        print('=====Sparsity metric=====')
        # print(sparse_train_array)
        # print(sparse_test_array)
        std_train_sparse = np.std(sparse_train_array)
        std_test_sparse = np.std(sparse_test_array)
        print('Training sparsity std')
        print(std_train_sparse)
        print('Testing sparsity std')
        print(std_test_sparse)

    # Average from all runs
    avg_train_acc = np.mean(train_acc_array) * 100
    avg_test_acc  = np.mean(test_acc_array) * 100
    std_train_acc = np.std(train_acc_array) * 100
    std_test_acc  = np.std(test_acc_array) * 100
    print('The average train accuracy is: {} +/ {}'.format(avg_train_acc, std_train_acc))
    print(' The average test accuracy is: {} +/ {}'.format(avg_test_acc, std_test_acc))

    # print('=====Entropy metric=====')
    # train_entropy, test_entropy = entropy(sdr_train_array, sdr_test_array)
    # print('Training entropy')
    # print(train_entropy)
    # print('Testing entropy')
    # print(test_entropy)


    print('=====Program has finished=====')