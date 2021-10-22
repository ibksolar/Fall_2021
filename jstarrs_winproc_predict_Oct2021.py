

import glob
import numpy as np
from scipy.io import loadmat,savemat
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model,load_model
from IPython.core.interactiveshell import InteractiveShell

import os
InteractiveShell.ast_node_interactivity = "all"


# Path to data
base_echo_path = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data'  # < == FIX HERE e.g os.path.join( os.getcwd(), echo_path )
echo_path = base_echo_path + 'Dec_block_len_45_Test_set191021'  #  < == FIX HERE e.g Full_block_len_45_280921_1530'


# Path to model
model_path = r'Y:\ibikunle\Python_Env\jstars_weight_21_15_filtered_image_july2021.h5'  #  < == FIX HERE

# Defaults
save_predictions = True
custom_normalize = False
plot_layers = False

#model2 = tf.keras.models.load_model('TBase.h5'); #loaded_model = tf.keras.models.load_model('new_weights_norm65_29_2.h5'); #loaded_model = tf.keras.models.load_model('../jstars_weight_21_15_july2021_resaved.h5')

# Load saved model and weights
loaded_model = tf.keras.models.load_model( model_path)      #tf.keras.models.load_model('full45x21_echo_Sept2021.h5')

# Load data
all_echo = glob.glob(echo_path + "/image/*.mat")
all_echo = sorted(all_echo)    
  
#Load layer e.g all_layer = glob.glob("jstars21_testset/layer/*.mat")
all_layer = glob.glob(echo_path + "/layer/*.mat")
all_layer = sorted(all_layer)  
      

filt_y = 45; filt_x = 15
conv_filter = np.zeros([filt_y, filt_x])
all_len = filt_y * filt_x


half_x = int((filt_x-1)/2)  
fill_val = filt_y//2

conv_filter[filt_y//2 ,filt_x//2] = 1

if half_x % 2 == 0:
     print('Filter column size is not odd, this might lead to unexpected results')
     
     
for idx in range(15): #len(all_echo)  range(5) range(550,570)
    data = loadmat(all_echo[idx]) 
    echo = data['echo_tmp']
    
    layer_data = loadmat(all_layer[idx])
    layer = layer_data['vec_layer'] 
    
    # predictions are the predictions using cumulative pred from model
    # prediction2 uses perfect ground truth knowledge to know where RowBlocks should start
    # predictions with nan replaces all pred==0 with nan (but interpolated values *not* nans are not used to compute the next rowBlock start)
    
    predictions,predictions2 = [],[]
    predictions_w_nan,predictions2_w_nan = [],[]
    raw_prediction = []

    predictions.append(  np.asarray(layer[0,:])  ) # initialize to surface location
    predictions2.append( np.asarray(layer[0,:])  )# initialize to surface location
    
    # Predictions with nan is the version of prediction where all zero prediction is set to NaN
    predictions_w_nan.append(  np.asarray(layer[0,:])  ) # initialize to surface location
    predictions2_w_nan.append( np.asarray(layer[0,:])  )# initialize to surface location
    raw_prediction.append( np.asarray(layer[0,:])  )# initialize to surface location
    
    results = {}
    num_rows,_  = echo.shape 
    mod_echo = np.concatenate ( (np.fliplr( echo[:,1:half_x+1]), echo, np.fliplr(echo[:,(-half_x-1):-1] )), axis = 1) # Mirror the edges
    
    if custom_normalize: 
        # Rarely ever use this
        mod_echo = ( mod_echo +65 )/36  # Convert log data back to linear scale
        mod_echo2 = 10**(mod_echo/10)
        mod_echo2 /= np.amax(mod_echo2)
        mod_echo /= 0.0011731572436752015      #np.amax(mod_echo) # normalize    
    
    next_row_block_start = layer[0,:]   
    next_row_block_start2 = layer[0,:]  
    
    Nt,Nx = echo.shape 
    
    # del data; del echo  # Delete some variables because of memory
    all_zero_count = 0
    count = 0
    
    while np.any( np.asarray(next_row_block_start) <= Nt-filt_y ) and all_zero_count < 15 : #
        predict, predict2  = [], []  # re-initialize "predict" for every layer
                     
        for iter_idx in range(Nx): # iterate through 
            row = int(next_row_block_start[iter_idx]) ;     row2 = int(next_row_block_start2[iter_idx]) 
            col = int(iter_idx);     
            predict.append(np.argmax( loaded_model.predict( (conv2( conv_filter, mod_echo [row:row+filt_y, col:col+filt_x], mode='same').T.ravel()).reshape(1,all_len))  )  )             
            predict2.append(np.argmax( loaded_model.predict( (conv2( conv_filter, mod_echo [row2:row2+filt_y, col:col+filt_x], mode='same').T.ravel()).reshape(1,all_len))  )  )             

        if np.any ( np.asarray(predict) == 0):
            
            # Interpolate to fill missing zeros
            ## Check if all prediction is zero ##
            if np.all( np.asarray(predict) == 0 ):
              all_zero_count +=1
              past_pred_zeros = True
              predict_intpd = fill_val
            
            else:                
              # Just some of the predictions have zeros 
              predict_intpd = pd.Series(predict)
              predict_intpd [predict_intpd == 0] = np.nan
              x= np.arange(predict_intpd.size)
              predict_intpd[np.isnan(predict_intpd)] = ( np.interp(x[np.isnan(predict_intpd)], x[np.isfinite(predict_intpd)],predict_intpd[np.isfinite(predict_intpd)])  ).astype('int')
              past_pred_zeros = False

        else:
            # No zeros in the predictions
          predict_intpd = predict
          past_pred_zeros = False
        
        ## Determine the next rowBlock start  
        next_row_block_start = next_row_block_start + predict_intpd        
        
        if count <= Nx-1:
            count +=1
            next_row_block_start2 = layer[count,:] 
        
        
        ## The predictions by the model for a layer may contain 0, those predictions are set to NaN and saved: predictions_w_nan is the real prediction 
        # However, for calculating the next rowBlock start, missing gaps should be interpolated
        temp = next_row_block_start.copy()
        
        if any( np.asarray(predict)==0  ):
            temp[ np.asarray(predict)==0 ] = np.nan      
        
        
        ## Cumulate predictions
        #============================================       

        predictions.append(next_row_block_start) # Interpolated and perfect prediction
        predictions2.append(predict2 + predictions2[-1] ) # Prediction using perfect ground truth           
        predictions_w_nan.append(temp + predictions[-1]) #Prediction with nans for predicted zeros        
        raw_prediction.append(predict) # raw predictions for each layer      
          
    if plot_layers:        
        fig, axes = plt.subplots(figsize=(15,30),dpi = 100);
        axes.imshow(echo,cmap='gray');
        axes.plot( np.arange(Nx),np.asarray(layer).T,'b-', np.arange(Nx),np.asarray(predictions_w_nan).T,'g--' );
        
#============================================
## Cumulate prediction and save predictions
#============================================
        
    fn ='echo_prediction'       
    results[fn] = predictions
    
    fn = 'gt_prediction'   # Predictions using perfect ground truth    
    results[fn] = predictions2#
    
    fn = 'prediction_w_nan'        
    results[fn] = predictions_w_nan
    
    fn = 'raw_prediction'            
    results[fn] = raw_prediction
    
    if save_predictions:
 
        fn = '/predictions_' +  os.path.basename(all_layer[idx]) # 'newest_predictions_echo%06d.mat'% (idx+1)  #os.path.splitext        
        if not os.path.isdir( echo_path+'/Predictions_folder' ):
            os.makedirs( echo_path+'/Predictions_folder' )
    
        save_path = echo_path +'/Predictions_folder' + fn
        savemat(save_path, results)
##    
##    del predictions,predictions2,results
##    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    