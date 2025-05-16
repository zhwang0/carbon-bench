# import os
import numpy as np


# output bands
TREE_BAND = ['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh']

# parameters
N_T = 12 
N_YEAR = 40
N_AGE = 15
N_OUT = len(TREE_BAND)

N_CONS = 138 # number of atmos features
N_FEA = N_CONS + N_OUT + 15 # add age info


