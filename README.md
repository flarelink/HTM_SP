# HTM_SP
Python program to create the Spatial Pooler from the Hierarchical Temporal Memory algorithm. This is a program to model the Spatial Pooler (SP) portion of the Hierarchical Temporal Memory (HTM) algorithm
developed by Jeff Hawkins and Numenta.

Usage: python2 SP_Main.py

Disclaimers:  -This project requires Python 2.
              -This project requires you to have the following datasets if you'd like to test each portion:
                -> MNIST
                -> Iris
                -> UCI Breast Cancer
                -> Fashion MNIST (I was given a binarized 32x32 dataset of this but at some point I'll make a
                                  bin_FASHION.py similar to bin_MNIST.py that will take the input csv of FASHION MNIST
                                  and create binarized versions of it.)


The SP's default hyperparameters are as follows:

number of columns               = 256  
number of proximal connections  = 31  
permanence threshold            = 0.5  
permanence increment            = 0.01  
permanence decrement            = -0.01  
minimum overlap score           = 2  
number of winners               = 40

boosting factor on overlap score= 3

speed of boost factors (T)      = 1000

Softmax classifier hyperparameters are as follows:
number of epochs  = 50
learning rate     = 0.001

I was able to achieve a little over 90% on MNIST, where it was 28x28 pixels (being the original MNIST),
by using the follow hyperparameters:
SP:
number of columns               = 1024
number of proximal connections  = 31
permanence threshold            = 0.5
permanence increment            = 0.01
permanence decrement            = -0.02
minimum overlap score           = 2
number of winners               = 40
boosting factor on overlap score= 3
speed of boost factors (T)      = 1000

Softmax:
number of epochs  = 100
learning rate     = 0.001

With these parameters I was able to achieve:
Training Accuracy: 92.418%
 Testing Accuracy: 90.660%


The default dataset is set to MNIST which is binarized down to 16x16. If you set the parameter --bin_MNIST to 28 then it will just recreate the original MNIST train and test csvs to 'MNIST_train_bin_784.csv' and 'MNIST_test_bin_784.csv'.

The scalar encoder in the encoders.py file is a modified version from Nupic's implementation but retains many of the same fundamentals to the original implementation. I used it primarily for testing purposes for the Iris and UCI Breast Cancer datasets. 
