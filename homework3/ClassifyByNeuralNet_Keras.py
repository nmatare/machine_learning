"""classify Human Activities by Feed-Forward Neural Network via Keras package

This script is designed to be run from the Command Line Terminal.

Please first set your desired settings in the USER-ADJUSTABLE GLOBAL CONSTANTS section in the script.

To run, in the Command Line Terminal, navigate to the same folder and issue command:
    (in [ ] are optional arguments)

    python <THIS_SCRIPT_FILE_NAME>.py [--data <DATA_FOLDER_PATH>]

where:

<DATA_FOLDER_PATH> is the data folder cloned from GitHub repository
https://github.com/ChicagoBoothML/DATA___UCI___HumanActivityRecognitionUsingSmartphones.

If <DATA_FOLDER_PATH> is not provided, the script will attempt to connect to the internet and download the above
repository onto a temporary folder, extract the necessary data and then delete the temporary folder.
"""

# Imports from Python v3
from __future__ import division, print_function

# Generic imports
from argparse import ArgumentParser
from cPickle import dump, load
from numpy import arange
from os import system
from random import sample
from sklearn.preprocessing import LabelBinarizer
from sys import path

# Keras imports
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

# Imports from other modules in same folder
from ParseData import parse_data

# Imports from other modules not in same folder
system('pip install --upgrade git+git://GitHub.com/ChicagoBoothML/Helpy --no-dependencies')
from ChicagoBoothML_Helpy.KerasTrainingMonitor import NeuralNetworkTrainingMonitor
from ChicagoBoothML_Helpy.Print import printflush, pprintflush

# Global constants
NB_HUMAN_ACTIVITES = 6
NB_EXAMPLE_FEATURES = 10


# **********************************************************************************************************************
# *** USER-ADJUSTABLE GLOBAL CONSTANTS *********************************************************************************
# **********************************************************************************************************************
LOAD_TRAINED_NEURAL_NETWORK_FROM_FILE_NAME = ''   # default=''; Give non-empty file name (NO EXTENSION) to load
SAVE_TRAINED_NEURAL_NETWORK_TO_FILE_NAME = ''   # default=''; Give non-empty file name (NO EXTENSION) to save to

NB_NEURAL_NETWORK_HIDDEN_UNITS = 100   #default=100

SGD_OPTIMIZER_LEARNING_RATE = .01   # default=.01
SGD_OPTIMIZER_LEARNING_RATE_DECAY_RATE = .0   # default=0, meaning no learning rate decay
SGD_OPTIMIZER_MOMENTUM_RATE = .9   # default=.9; 0 means no momentum
SGD_OPTIMIZER_NESTEROV_MOMENTUM_YESNO = True   # default=True; using Nesterov momentum usually speeds up learning

NB_TRAIN_EPOCHS = 30   # default=30
TRAIN_MINI_BATCH_SIZE = 300   # default=300
VALIDATION_DATA_PROPORTION = .2   # default=.2; this is proportion of Training Data held out for validation
# **********************************************************************************************************************


# Parse the relevant data sets
if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data', help="Path to cloned repository containing Human Activities data set")
    args = vars(arg_parser.parse_args())
    printflush('Command-Line Arguments:')
    pprintflush(args)
else:
    args = dict.fromkeys(('data',))

data_path = args['data']
if data_path:
    data = parse_data(data_path)
else:
    data = parse_data()

label_binarizer = LabelBinarizer()
label_binarizer.fit(arange(NB_HUMAN_ACTIVITES))
X_names = data['X_names']
train_X = data['train_X']
train_y = data['train_y']
train_y_binary_matrix = label_binarizer.transform(train_y)
test_X = data['test_X']
test_y = data['test_y']
test_y_binary_matrix = label_binarizer.transform(test_y)
y_class_labels = data['y_class_labels']


# Get some basic counts on the data sets
nb_X_features = len(X_names)
nb_train_cases = len(train_X)
nb_test_cases = len(test_X)
printflush('No. of Input Features X = %s' % '{:,}'.format(nb_X_features))
printflush('No. of Class Labels y = %s' % '{:,}'.format(NB_HUMAN_ACTIVITES))
printflush('No. of Train Cases = %s' % '{:,}'.format(nb_train_cases))
printflush('No. of Test Cases = %s' % '{:,}'.format(nb_test_cases))

# View the names of the X features
printflush('Example of 10 Input Features X:')
pprintflush(sample(X_names, NB_EXAMPLE_FEATURES))


# Load or Create Neural Network
if LOAD_TRAINED_NEURAL_NETWORK_FROM_FILE_NAME:
    load_file_name = LOAD_TRAINED_NEURAL_NETWORK_FROM_FILE_NAME + '.pickle'
    printflush('\nLoading Previously-Trained Feed-Forward Neural Network (FFNN) from %s... ' % load_file_name.upper(),
               end='')
    ffnn = load(open(load_file_name, 'rb'))
else:
    # Create Feed-Forward Neural Network (FFNN)
    # (doc: http://keras.io/models/#sequential)
    printflush('\nCreating Feed-Forward Neural Network (FFNN)... ', end='')

    ffnn = Sequential()
    ffnn.add(Dense(input_dim=nb_X_features,
                   output_dim=NB_NEURAL_NETWORK_HIDDEN_UNITS,
                   init='uniform'))
    ffnn.add(Activation('tanh'))
    ffnn.add(Dense(input_dim=NB_NEURAL_NETWORK_HIDDEN_UNITS,
                   output_dim=NB_HUMAN_ACTIVITES,
                   init='uniform'))
    ffnn.add(Activation('softmax'))
printflush('done!\n')


# Set FFNN's Loss Function & Optimizer
printflush('\nCompiling FFNN with Objective Loss Function & Optimization Method... ', end='')
stochastic_gradient_descent_optimizer = SGD(lr=SGD_OPTIMIZER_LEARNING_RATE,
                                            decay=SGD_OPTIMIZER_LEARNING_RATE_DECAY_RATE,
                                            momentum=SGD_OPTIMIZER_MOMENTUM_RATE,
                                            nesterov=SGD_OPTIMIZER_NESTEROV_MOMENTUM_YESNO)
ffnn.compile(loss='categorical_crossentropy',
             optimizer=stochastic_gradient_descent_optimizer)
printflush('done!\n')


# Initiate FFNN Training Monitor to keep track of training progress
ffnn_training_history = NeuralNetworkTrainingMonitor(
    plot_title='Neural Network Learning Curves: Human Activity Recognition')


# Train FFNN
ffnn.fit(X=train_X.values,
         y=train_y_binary_matrix,
         nb_epoch=NB_TRAIN_EPOCHS,
         batch_size=TRAIN_MINI_BATCH_SIZE,
         show_accuracy=True,
         validation_split=VALIDATION_DATA_PROPORTION,
         verbose=0,   # no need to log output to the terminal because we already have the live plot
         callbacks=[ffnn_training_history],
         shuffle=True)


# Obtain, and save (if instructed), the best trained FFNN
ffnn = ffnn_training_history.best_model
if SAVE_TRAINED_NEURAL_NETWORK_TO_FILE_NAME:
    save_file_name = SAVE_TRAINED_NEURAL_NETWORK_TO_FILE_NAME + '.pickle'
    printflush('\nSaving Trained FFNN to %s... ' % save_file_name.upper(), end='')
    dump(ffnn, open(save_file_name, 'wb'))
    printflush('done!\n')


# Evaluate trained FFNN on Test Data
printflush('\nEvaluating Trained FFNN on Test Data...')
test_evaluation = ffnn.evaluate(X=test_X.values,
                                y=test_y_binary_matrix,
                                show_accuracy=True,
                                verbose=0)
printflush('Test Set Loss = %s' % '{:.3g}'.format(test_evaluation[0]))
printflush('Test Set Accuracy = %s%%' % '{:.2f}'.format(100. * test_evaluation[1]))
