import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn as skl
import cv2
from metrics import sparseness

def SP_init(train_data, test_data, n_cols, n_proxim_con, perm_thresh, perm_inc, perm_dec, min_overlap, n_winners, beta_boost, T_boost_speed, verbose):
    """
    Spatial Pooler Algorithm of Hierarchical Temporal Memory
    
    :param train_data   : input training data
    :param test_data    : input testing data
    :param n_cols       : number of columns
    :param n_proxim_con : number of proximal connections
    :param perm_thresh  : permanence threshold
    :param perm_inc     : permanence increment factor
    :param perm_dec     : permanence decrement factor
    :param min_overlap  : minimum overlap score
    :param n_winners    : number of winners from winner-take-all
    :param beta_boost   : boosting factor
    :param T_boost_speed: time steps for boosting
    :param verbose      : verbosity flag
    
    :return: sdr_train_array       : output sdr train array from input training data
                 sdr_test_array    : output sdr test array from input testing data   
                 sparse_train_array: sparse array to determine sparsity metric of sdr_train_array
                 sparse_test_array : sparse array to determine sparsity metric of sdr_test_array   
    """

    # Load MNIST data input_size (aka nn) of 256 or 1024
    input_size = len(train_data[1])
    # print(len(train_data[1]))

    # Initialize synapses and permanence arrays
    syn_index = np.random.randint(0, input_size, (n_cols, n_proxim_con))
    syn_array = np.zeros((n_cols, input_size), dtype=int)
    # syn_array[syn_index] = 1

    # Synapses array
    for i in xrange(n_cols):
        syn_array[i, syn_index[i]] = 1

    # syn_array = np.random.randint(0, 2, (n_cols, input_size))
    perm_array = np.random.uniform(0, 1, (n_cols, input_size))
    perm_array = syn_array * perm_array

    # Initialize empty SDR array
    # overlap_scores = np.zeros([1, n_cols])
    sdr_train_array = np.zeros((len(train_data), n_cols), dtype=int)
    sdr_test_array = np.zeros((len(test_data), n_cols), dtype=int)

    # Initialize empty boosting arrays; time-averaged activation level
    time_avg_act = np.zeros([1, n_cols])
    prev_time_avg_act = np.zeros([1, n_cols])
    boosting = np.ones([1, n_cols])

    # Initialize metric arrays
    sparse_train_array = np.zeros(([len(train_data), 1]))
    sparse_test_array = np.zeros(([len(test_data), 1]))

    # Main code
    train_en = True
    for epoch in xrange(0, 2):
        input_set = train_data
        if train_en == False:
            input_set = test_data

        for iter in xrange(0, len(input_set)):
            # Calculate overlap scores
            overlap_scores = np.dot((syn_array * (perm_array >= perm_thresh)), input_set[iter, :].transpose()) \
                             * boosting

            # Initialize SDR (activations of cols)
            sdr = np.zeros(([1, n_cols]), dtype=int)

            # Select the winners
            for i in xrange(n_winners):
                win_val = np.max(overlap_scores)
                win_index = np.argmax(overlap_scores)
                if(win_val >= min_overlap):
                    sdr[0, win_index] = 1
                    overlap_scores[0, win_index] = 0

            #num_wins = sum(sdr)
            #print('This is num_wins')
            #print(num_wins)

            # Calculating activation level current and then previous, a_bar(t) and a_bar(t-1)
            if iter >= T_boost_speed:
                time_avg_act = ((T_boost_speed - 1) * prev_time_avg_act + sdr) / T_boost_speed
            prev_time_avg_act = time_avg_act

            # Calculating mini column neighborhood
            recent_act = (1/abs(n_cols)) * np.sum(time_avg_act)

            # Calculate boosting for next time
            boosting = np.exp(-beta_boost * (time_avg_act - recent_act))

            if(train_en == True):
                # Update permanence values for learning -> Hebbian learning
                z = sdr.transpose() * syn_array
                polar_input = np.copy(input_set[iter, :])
                polar_input[polar_input == 1] = perm_inc
                polar_input[polar_input == 0] = perm_dec
                delta_perm = polar_input * z
                perm_array = perm_array + delta_perm
                perm_array[perm_array > 1] = 1
                perm_array[perm_array < 0] = 0

            # Add SDR to array and calculate metrics
            # Metrics include: sparseness
            if train_en == False:
                sdr_test_array[iter, :] = sdr
                sparse_test_array[iter, 0] = sparseness(n_cols, sdr)
            else:
                sdr_train_array[iter, :] = sdr
                sparse_train_array[iter, 0] = sparseness(n_cols, sdr)

            # You are set!!!

            if(verbose):
                # Plot the image
                sdr_image = np.reshape(sdr, (16, 16))

                # Plot the image
                image = np.reshape(train_data[iter, :], (32, 32))
                plt.figure(1)
                plt.subplot(211)
                plt.imshow(image, cmap='gray_r')
                plt.title('Train')

                # Plot the sdr
                plt.subplot(212)
                plt.imshow(sdr_image, cmap='gray_r')
                plt.title('SDR')
                plt.tight_layout()
                plt.show()
                if(iter % 10 == 0):
                    print(iter)

        train_en = False
    return sdr_train_array, sdr_test_array, sparse_train_array, sparse_test_array

