# -*- coding: utf-8 -*-
"""
Basic script to collect stat about 
the dataset
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.io import loadmat
# from itertools import compress
import csv
import random
import shutil

echo_dir = r'Y:\ibikunle\ct_user_tmp\JSTARRS2021_Sep21\snow\2012_Greenland_P3\frames_001_243_20120330_04'

all_echo = glob.glob(echo_dir + "/image/*.mat")
all_echo = sorted(all_echo)

all_layer = glob.glob(echo_dir + "/layer/*.png")
all_layer = sorted(all_layer)

all_layer_bin = glob.glob(echo_dir + "/layer_bin/*.png")
all_layer_bin = sorted(all_layer_bin)

# Pre-allocate
frame_idx = []
num_layers = []
num_valid_layers = []
layers_with_gaps = []
idx_layers_with_gaps = []
total_gaps = []

create_sheet = False

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

Contains_gaps =[True if iter>0 else False for iter in total_gaps] 
   
pd_dict = {"frame_idx": frame_idx,
           "num_layers": num_layers,
           "num_valid_layers": num_valid_layers,
           "layers_with_gaps": layers_with_gaps,
           "idx_layers_with_gaps": idx_layers_with_gaps,
           "total_gaps": total_gaps, 
           "Contains_gaps": Contains_gaps
           }

df = pd.DataFrame(pd_dict)
writer = pd.ExcelWriter(echo_dir +'/stat_output.xlsx')
# write dataframe to excel

if create_sheet:
    df.to_excel(writer)
    writer.save()

# gaps_idx, = np.where(Contains_gaps)
# bad_frames = list(compress(frame_idx,Contains_gaps))
# good_frames = list( compress(frame_idx,np.invert(Contains_gaps)) )

bad_frames = df.loc[df['Contains_gaps']==True, 'frame_idx' ]
good_frames = df.loc[df['Contains_gaps']==False, 'frame_idx' ]


train_good_frames = random.sample( list(good_frames), round(0.7*len(good_frames))  )
test_good_frames = [i for i in good_frames if i not in train_good_frames]

train_bad_frames = random.sample( list(bad_frames), round(0.7*len(bad_frames))  )
test_bad_frames = [i for i in bad_frames if i not in train_bad_frames]

# Check if no file is skipped
np.testing.assert_array_equal(len(good_frames), len(train_good_frames)+len(test_good_frames), err_msg="They are not equal" )
np.testing.assert_array_equal(len(bad_frames), len(train_bad_frames)+len(test_bad_frames), err_msg="They are not equal" )


# Folders and path 
train_path = '\\Train_set'
test_path = '\\Test_set' 

image_src_path = echo_dir + "\\image\\" 
image_train_dst_path = echo_dir + train_path + "\\image" 
image_test_dst_path = echo_dir + test_path + "\\image" 

layer_src_path = echo_dir + "\\layer\\" 
layer_train_dst_path = echo_dir + train_path + "\\layer\\"
layer_test_dst_path = echo_dir + test_path + "\\layer\\"

layer_bin_src_bin_path = echo_dir +"\\layer_bin\\"; 
layer_bin_train_dst_path = echo_dir + train_path + "\\layer_bin\\"
layer_bin_test_dst_path = echo_dir + test_path + "\\layer_bin\\"

# files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]

## Check paths if they exist and create if they don't

if not os.path.exists(echo_dir + train_path):
    os.mkdir(echo_dir + train_path)

if not os.path.exists(echo_dir + test_path):
    print(f' Destination folders{echo_dir + test_path} do not exist, creating paths... ')
    os.mkdir(echo_dir + test_path)
    
if not os.path.exists(image_train_dst_path):
    os.mkdir(image_train_dst_path)
if not os.path.exists(layer_train_dst_path):
    os.mkdir(layer_train_dst_path)
if not os.path.exists(layer_bin_train_dst_path):
    os.mkdir(layer_bin_train_dst_path)
    
if not os.path.exists(image_test_dst_path):
    os.mkdir(image_test_dst_path)
if not os.path.exists(layer_test_dst_path):
    os.mkdir(layer_test_dst_path)
if not os.path.exists(layer_bin_test_dst_path):
    os.mkdir(layer_bin_test_dst_path)
        


#########################################################################
# Copy (Move) training GOOD files
for f in train_good_frames:
    base_fn = os.path.splitext(f)[0]
   
    # Copy image (.mat and.tiff)
    if os.path.isfile(image_src_path + f):
        shutil.copy(image_src_path + f, image_train_dst_path)
        shutil.copy(image_src_path + base_fn +'.tiff', image_train_dst_path)
    else:
        print(f' Could not copy Train_good_image {f} to {image_train_dst_path}')          
   
     
    # Copy Layer
    layer_fn = base_fn.replace("image","layer") 
    if os.path.isfile(layer_src_path + layer_fn + '.png'):
        shutil.copy(layer_src_path + layer_fn + '.png', layer_train_dst_path)
    else:
        print(f' Could not copy Layer_image {f} to {layer_train_dst_path}')        
    
       
    # Copy Layer_bin
    layer_bin_fn = base_fn.replace("image","layer_binary") 
    if os.path.isfile(layer_bin_src_bin_path + layer_bin_fn + '.png'):
        shutil.copy(layer_bin_src_bin_path + layer_bin_fn + '.png', layer_bin_train_dst_path)
    else:
        print(f' Could not copy Layerbin_image {f} to {image_train_dst_path}')
        
#########################################################################
                # TRAIN FILES
#########################################################################

# Copy (Move) Train BAD files
for f in train_bad_frames:
    base_fn = os.path.splitext(f)[0]
   
    # Copy image (.mat and.tiff)
    if os.path.isfile(image_src_path + f):
        shutil.copy(image_src_path + f, image_train_dst_path)
        shutil.copy(image_src_path + base_fn +'.tiff', image_train_dst_path)
    else:
        print(f' Could not copy train_bad_image {f} to {image_train_dst_path}')          
   
     
    # Copy Layer
    layer_fn = base_fn.replace("image","layer") 
    if os.path.isfile(layer_src_path + layer_fn + '.png'):
        shutil.copy(layer_src_path + layer_fn + '.png', layer_train_dst_path)
    else:
        print(f' Could not copy train_bad_layer {f} to {layer_train_dst_path}')        
    
       
    # Copy Layer_bin
    layer_bin_fn = base_fn.replace("image","layer_binary") 
    if os.path.isfile(layer_bin_src_bin_path + layer_bin_fn + '.png'):
        shutil.copy(layer_bin_src_bin_path + layer_bin_fn + '.png', layer_bin_train_dst_path)
    else:
        print(f' Could not copy train_bad_layer_bin {f} to {image_train_dst_path}')


###########################################################################
                    # TEST FILES
#########################################################################
# Copy (Move) TEST GOOD files
for f in test_good_frames:
    base_fn = os.path.splitext(f)[0]
   
    # Copy image (.mat and.tiff)
    if os.path.isfile(image_src_path + f):
        shutil.copy(image_src_path + f, image_test_dst_path)
        shutil.copy(image_src_path + base_fn +'.tiff', image_test_dst_path)
    else:
        print(f' Could not copy {f} to {image_train_dst_path}')          
   
     
    # Copy Layer
    layer_fn = base_fn.replace("image","layer") 
    if os.path.isfile(layer_src_path + layer_fn + '.png'):
        shutil.copy(layer_src_path + layer_fn + '.png', layer_test_dst_path)
    else:
        print(f' Could not copy {f} to {layer_train_dst_path}')      
         
    # Copy Layer_bin
    layer_bin_fn = base_fn.replace("image","layer_binary") 
    if os.path.isfile(layer_bin_src_bin_path + layer_bin_fn + '.png'):
        shutil.copy(layer_bin_src_bin_path + layer_bin_fn + '.png', layer_bin_test_dst_path)
    else:
        print(f' Could not copy {f} to {image_train_dst_path}')
        
#########################################################################

# Copy (Move) TEST BAD files
for f in test_bad_frames:
    base_fn = os.path.splitext(f)[0]
   
    # Copy image (.mat and.tiff)
    if os.path.isfile(image_src_path + f):
        shutil.copy(image_src_path + f, image_test_dst_path)
        shutil.copy(image_src_path + base_fn +'.tiff', image_test_dst_path)
    else:
        print(f' Could not copy {f} to {image_train_dst_path}')          
   
     
    # Copy Layer
    layer_fn = base_fn.replace("image","layer") 
    if os.path.isfile(layer_src_path + layer_fn + '.png'):
        shutil.copy(layer_src_path + layer_fn + '.png', layer_test_dst_path)
    else:
        print(f' Could not copy {f} to {layer_train_dst_path}')        
    
       
    # Copy Layer_bin
    layer_bin_fn = base_fn.replace("image","layer_binary") 
    if os.path.isfile(layer_bin_src_bin_path + layer_bin_fn + '.png'):
        shutil.copy(layer_bin_src_bin_path + layer_bin_fn + '.png', layer_bin_test_dst_path)
    else:
        print(f' Could not copy {f} to {image_train_dst_path}')

###########################################################################


# Create and save names used good_frames and bad_frames
train_good_frames.insert(0,"train_good_frames")
train_bad_frames.insert(0,"train_bad_frames")
test_good_frames.insert(0, "test_good_frames")
test_bad_frames.insert(0,"test_bad_frames")

df2 = [train_good_frames, train_bad_frames, test_good_frames, test_bad_frames  ]

with open(echo_dir +'/train_test_frames_output.csv', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(df2)








