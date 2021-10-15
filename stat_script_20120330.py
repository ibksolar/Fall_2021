# -*- coding: utf-8 -*-
"""
Basic script to collect stat about 
the dataset
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.io import loadmat,savemat

echo_dir = r'Y:\ibikunle\ct_user_tmp\JSTARRS2021_Sep21\snow\2012_Greenland_P3\frames_001_243_20120330_04'

all_echo = glob.glob(echo_dir + "/image/*.mat")
all_echo = sorted(all_echo)

# Pre-allocate
frame_idx = []
num_layers = []
num_valid_layers = []
layers_with_gaps = []
idx_layers_with_gaps = []
total_gaps = []

for echo_idx in range(len(all_echo)):
    curr_echo = loadmat(all_echo[echo_idx])
    curr_layer = np.asarray( curr_echo["layer"] )
    
    temp = np.all( np.isnan(curr_layer), axis=1 )
    temp2 = curr_layer[~temp]
    
    # Update accumulators
    frame_idx.append(os.path.basename(all_echo[echo_idx]))
    num_layers.append( curr_layer.shape[0])
    num_valid_layers.append( sum(~temp))
    layers_with_gaps.append( np.where(np.any(np.isnan(temp2),axis = 1))[0].tolist() )    
    idx_layers_with_gaps.append( list(zip(*np.where(np.isnan(temp2) == 1))) )
    total_gaps.append( np.sum(np.isnan(temp2)) )
    
pd_dict = {"frame_idx": frame_idx,
           "num_layers": num_layers,
           "num_valid_layers": num_valid_layers,
           "layers_with_gaps": layers_with_gaps,
           "idx_layers_with_gaps": idx_layers_with_gaps,
           "total_gaps": total_gaps           
           }

df = pd.DataFrame(pd_dict)
writer = pd.ExcelWriter(echo_dir +'/stat_output.xlsx')
# write dataframe to excel
df.to_excel(writer)
writer.save()