import os
import numpy as np


global TREE_BAND
global N_OUT
global N_CONS
global N_FEA
global N_T
global Y_MEAN
global Y_STD

# output bands
TREE_BAND = ['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh']

# parameters
N_OUT = len(TREE_BAND)
N_CONS = 135 # number of atmos features
N_FEA = N_CONS + N_OUT + 15 # add age info
N_T = 12 
N_YEAR = 40
N_AGE = 8
BATCH_SIZE = 600

# scaler
# read scaler


# get output (tree) mean and std (init and output tree are considered as same)
y_scaler = np.load(os.path.join('configs', 'Y_SCALER.npy'))
Y_MEAN = y_scaler[0]
Y_STD = y_scaler[1]

