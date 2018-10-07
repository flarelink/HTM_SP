"""
Author: Humza Syed
Description: File containing encoder(s) for usage in HTM

Nearly all code utilized for this file was from:
https://github.com/numenta/nupic/blob/master/src/nupic/encoders/scalar.py 
Some modifications were made for my own experiments, however I am putting the
Numenta disclaimer as they should receive most of the credit for the code

# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/

"""

import numpy as np

def scalar_encoder(input_data_array, n, w, periodic, clipInput):
    """
    Scalar encoder to take input integer/float values and represent them in an output array of 0s and 1s

    :param input_data_array:
    :param n:                   # of bits for output representation, should be greater than w
    :param w:                   # of bits used for encoding, must be odd value to ensure centering problem doesn't occur
    :param periodic:            True or False depending on periodic, ex: days of week are periodic
    :param clipInput:           True/False flag meant to determine if input should be clipped if below/above min/max vals
    :return: output array representing inputs through 0s and 1s

    DISCLAIMER from Humza: I haven't tested periodic being equal to True

    example usage for breast cancer dataset:
    DISCLAIMERS from Humza: - CONVERTING ALL '?'s IN UCI BC TO VALUES OF 1
                            - CONVERTING ALL LABELS 2 --> 0 AND 4 --> 1

    input_data_array = training/test data without IDs or labels
    n = 1024
    w = 39
    periodic = False
    clipInput = False


    """

    # converting input data to floats
    input_data_array = input_data_array.astype(float)

    # setting min and max vals
    min_vals = np.amin(input_data_array, axis=0)
    max_vals = np.amax(input_data_array, axis=0)

    # Checks
    if(min_vals.any() == None or max_vals.any() == None):
        raise ValueError('min and max vals should be real values')
    for i in xrange(len(min_vals)):
        if(min_vals[i] >= max_vals[i]):
            raise ValueError('min_vals should not be greater than or equal to max_vals')
    if(w >= n):
        raise ValueError('w should not be greater than n')
    if(w % 2 == 0):
        raise ValueError('The value of w (act bits) should be odd to avoid centering problem')
    if(input_data_array.any() == None):
        raise ValueError('The input array is invalid')

    """
    Initialization of variables
    """
    # Determine number of parameters and total number of input examples being passed in
    parameters = len(input_data_array[0])
    total_num_input_data = len(input_data_array)

    # space that each parameter will use to be concatenated to represent output rep from input data
    # through the division the number of bits can be determined to represent a single parameter
    rep_space = n // parameters  # for 1024 and 9 parameters it will output 113 adding up to 1017
    if (rep_space <= w):
        ValueError('The value of w (act bits) should be decreased for representation space')

    """ 
    Define range, half the width of activated bits, and padding -
    from NuPIC: padding = For non-periodic inputs, padding is the number of bits "outside" the range,
    on each side. I.e. the representation of min_vals is centered on some bit, and
    there are "padding" bits to the left of that centered bit; similarly with
    bits to the right of the center bit of max_vals
    """
    halfwidth = (w - 1) / 2
    if (periodic == True):
        padding = 0
    else:
        padding = halfwidth

    # from NuPIC:  nInternal represents the output area excluding the possible padding on each side
    # nInternal = n - 2 * padding
    nInternal = rep_space - 2 * padding

    # Initializing numpy arrays
    # create empty array for output representation
    output_rep_array = np.zeros((total_num_input_data, n), dtype=float)

    # define internal range row vector
    rangeInternal = np.zeros((1, parameters), dtype=float)

    # define resolution row vector
    resolution = np.zeros((1, parameters), dtype=float)

    # define range
    _range = np.zeros((1, parameters), dtype=float)

    # define radius
    radius = np.zeros((1, parameters), dtype=float)

    # Set all variables
    for c in xrange(0, parameters):
        # # Normalize the input
        # input_data_array[:, c] = input_data_array[:, c] / float(max_vals[c])

        # Create row vector for rangeInternal
        rangeInternal[0, c] = float(max_vals[c] - min_vals[c])

        """
        from NuPIC: resolution = Two inputs separated by greater than, or equal to the
                    resolution are guaranteed to have different
                    representations.
        """
        if not periodic:
            resolution[0, c] = float(rangeInternal[0, c]) / (rep_space - w)
        else:
            resolution[0, c] = float(rangeInternal[0, c]) / (rep_space)

        """
        from NuPIC: radius = Two inputs separated by more than the radius have
                    non-overlapping representations. Two inputs separated by less
                    than the radius will in general overlap in at least some of
                    their bits. You can think of this as the radius of the input
        """
        radius[0, c] = w * resolution[0, c]

        # determine range
        if (periodic == True):
            _range = rangeInternal[0, c]
        else:
            _range = rangeInternal[0, c] + resolution[0, c]


    def _getFirstOnBit(input_row, min_vals, max_vals):
        """
        from NuPIC: Return the bit offset of the first bit to be set in the encoder output.
        For periodic encoders, this can be a negative number when the encoded output
        wraps around.
        """

        centerbin = np.zeros((1, len(input_row)))
        minbin = np.zeros((1, len(input_row)))

        for col in xrange(len(input_row)):
            if input_row[col] < min_vals[col]:
                # Don't clip periodic inputs. Out-of-range input is always an error
                if clipInput and not periodic:
                    print "Clipped input %.2f to min_vals %.2f" % (input_row[col], min_vals[col])
                    input_row[col] = min_vals[col]
                else:
                    raise Exception('input (%s) less than range (%s - %s)' %
                                    (str(input_row[col]), str(min_vals[col]), str(max_vals[col])))

            if periodic:
                # Don't clip periodic inputs. Out-of-range input is always an error
                if input_row[col] >= max_vals[col]:
                    raise Exception('input (%s) greater than periodic range (%s - %s)' %
                                    (str(input_row[col]), str(min_vals[col]), str(max_vals[col])))
            else:
                if input_row[col] > max_vals[col]:
                    if clipInput:
                        print "Clipped input %.2f to max_vals %.2f" % (input_row[col], max_vals[col])
                        input_row[col] = max_vals[col]
                    else:
                        raise Exception('input (%s) greater than range (%s - %s)' %
                                        (str(input_row[col]), str(min_vals[col]), str(max_vals[col])))

            if periodic:
                centerbin[0, col] = int((input_row[col] - min_vals[col]) * nInternal / _range[0, col]) + padding
            else:
                centerbin[0, col] = int(((input_row[col] - min_vals[col]) + resolution[0, col] / 2) / resolution[0, col]) + padding

            # We use the first bit to be set in the encoded output as the bucket index
            minbin[0, col] = centerbin[0, col] - halfwidth

        return minbin



    ### MAIN ENCODING ###
    def encodeIntoArray(input_row, rep_space, min_vals, max_vals, output):
        """
            All comments from NuPIC
        """

        maxbin = np.zeros((1, len(input_row)))
        indexing = 0

        # Get the bucket index to use
        bucketIdx = _getFirstOnBit(input_row, min_vals, max_vals)#[0]

        if bucketIdx is None:
            # None is returned for missing value
            output[0:rep_space] = 0  # TODO: should all 1s, or random SDR be returned instead?

        else:
            # The bucket index is the index of the first bit to set in the output
            minbin = bucketIdx

            for m in xrange(len(input_row)):
                maxbin[0, m] = minbin[0, m] + 2 * halfwidth
                if periodic:
                    # Handle the edges by computing wrap-around
                    if maxbin[0, m] >= rep_space:
                        bottombins = maxbin[0, m] - rep_space + 1
                        output[:bottombins] = 1
                        maxbin[0, m] = rep_space - 1
                    if minbin[0, m] < 0:
                        topbins = -minbin[0, m]
                        output[rep_space - topbins:rep_space] = 1
                        minbin[0, m] = 0

                assert minbin[0, m] >= 0
                assert maxbin[0, m] < rep_space

                minbin = minbin.astype(int)
                maxbin = maxbin.astype(int)
                # set the output (except for periodic wraparound)
                output[indexing + minbin[0, m]:indexing + maxbin[0, m] + 1] = 1
                indexing += rep_space

        # Debug the decode() method
        #print "output:",
        #print(output)

    for i in xrange(total_num_input_data):
        encodeIntoArray(input_data_array[i, :], rep_space, min_vals, max_vals, output_rep_array[i, :])
        # print(output_rep_array)

    return output_rep_array
