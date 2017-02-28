import pip
import urllib2
import getpass
import os
import pandas as pd

# Setup environment 

os.environ['COLUMNS'] = "200"
dir = '/home/' + getpass.getuser() + '/Documents/Education/Chicago_Booth/Classes/41204_Machine_Learning/machine_learning/homework3/'
os.chdir(dir)

# Read in data

# https://www.tensorflow.org/get_started/tflearn

X_train = pd.read_csv('X_train.csv') 
Y_train = pd.read_csv('X_train.csv')

X_test = pd.read_csv('X_test.csv')
Y_test = pd.read_csv('Y_test.csv')

#