# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:02:21 2021

@author: i368o351
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

decimate_in_fn = False

echo_dir = r'Y:\ibikunle\ct_user_tmp\JSTARRS2021_final290621\snow\2012_Greenland_P3\frames_001_243_20120330_04'

list_src = r'Y:\ibikunle\ct_user_tmp\JSTARRS2021_final290621\snow\2012_Greenland_P3\frames_001_243_20120330_04\train_test_frames_output.csv'

df = pd.read_csv(list_src, header=None)
train_good_frames =  df.iloc[0,2:].dropna(how='all') # img_src +
train_bad_frames =  df.iloc[1,2:].dropna(how='all')
test_good_frames =  df.iloc[2,2:].dropna(how='all')
test_bad_frames =   df.iloc[3,2:].dropna(how='all')


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
# TO DO: Rewrite this as a function instead

for f in train_good_frames:
    
    if decimate_in_fn:
        dec_idx = f.find('.mat')         
        f = f[:dec_idx] + '_dec' + f[dec_idx:]
      
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
    
    if decimate_in_fn:
        dec_idx = f.find('.mat')         
        f = f[:dec_idx] + '_dec' + f[dec_idx:]
        
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
    
    if decimate_in_fn:
        dec_idx = f.find('.mat')         
        f = f[:dec_idx] + '_dec' + f[dec_idx:]
        
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
    
    if decimate_in_fn:
        dec_idx = f.find('.mat')         
        f = f[:dec_idx] + '_dec' + f[dec_idx:]
    
    
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