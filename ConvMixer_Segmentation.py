# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 15:55:12 2021

Train and test using newly created Train and Test set

@author: i368o351
"""

from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
import tensorflow_addons as tfa

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau

from scipy.io import loadmat
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"



print("Packages Loaded")

#========================================================================
# ==================LOAD DATA =========================================
#========================================================================

# Might need to revise this: this might not be the best approach considering memory

base_dir = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Dec_block_len_45_181021_1828'

input_dir = "/image/"
target_dir = "/raster_dir/"

# Confirm path is correct
if os.path.isdir(base_dir+target_dir):
    print('Target path is okay')
else:
    print('Target path is broken: please fix...')


input_img_paths = sorted( os.listdir (base_dir+ input_dir) ) 
target_img_paths = sorted( os.listdir(base_dir + target_dir) ) 

one_path = target_img_paths[10]  
one_data = loadmat(base_dir+ target_dir + '/' + one_path)
one_data = one_data['raster']

num_classes = 1 # int(max(np.unique(one_data)))
image_x,image_y = one_data.shape
img_size = one_data.shape

# Hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32 #128
num_epochs = 20
num_channels = 1

train_samples = round(0.85* len(input_img_paths))       
val_samples = round(0.1* len(input_img_paths)) # 500
test_samples = len(input_img_paths) - train_samples - val_samples

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:train_samples] # input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:train_samples] # target_img_paths[:-val_samples]

val_input_img_paths = input_img_paths[train_samples:train_samples+val_samples+1] # input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[train_samples:train_samples+val_samples+1]



# # Prepare tf.data.Dataset objects
# def make_datasets(images, labels, is_train=False):
#     dataset = tf.data.Dataset.from_tensor_slices((images, labels))
#     if is_train:
#         dataset = dataset.shuffle(batch_size * 10)
#     dataset = dataset.batch(batch_size)
#     if is_train:
#         dataset = dataset.map(
#             lambda x, y: (data_augmentation(x), y), num_parallel_calls=auto
#         )
#     return dataset.prefetch(auto)

# # AUGMENTATION
# image_x,image_y = x_train.shape[1:]
# img_size = (image_x,image_y)

# auto = tf.data.AUTOTUNE
# data_augmentation = Sequential(
#     [layers.RandomCrop(image_x, image_y), layers.RandomFlip("vertical"),],
#     name="data_augmentation",
# )

# Echo_Load_Train_Test function
class Echo_Load_Train_Test(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths,base_dir = base_dir ,input_dir = input_dir,target_dir = target_dir, num_classes = num_classes):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.base_dir = base_dir
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.num_classes = num_classes

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        # x = np.zeros((self.batch_size,) + self.img_size , dtype="uint8")
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img_path = base_dir + input_dir + path
            img = loadmat(img_path)
            img = img['echo_tmp']
            img[np.isnan(img)] = 0
            
            if np.all(img<=1):
                x[j] = np.expand_dims( img, 2) # Normalize /255
            else:
                x[j] = np.expand_dims( img/255, 2)

        # y = np.zeros((self.batch_size,) + self.img_size , dtype="uint8")    
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            target_path = base_dir + target_dir + path
            target = loadmat(target_path)
            target = target['raster'] # raster, semantic_seg
            target[np.isnan(target)] = 0
            target = ( np.array(target, dtype=bool) ) #,dtype=bool                        
            y[j] = np.expand_dims( target, 2 )
        # y = tf.keras.utils.to_categorical(y, num_classes)
        return x, y  

# Create training and testing data
train_dataset = Echo_Load_Train_Test(batch_size, img_size, train_input_img_paths, train_target_img_paths,num_classes)
val_dataset = Echo_Load_Train_Test(batch_size, img_size, val_input_img_paths, val_target_img_paths,num_classes)

if test_samples > 1:
    test_input_img_paths = input_img_paths[-test_samples:] # input_img_paths[-val_samples:]
    test_target_img_paths = target_img_paths[-test_samples:]
    test_dataset = Echo_Load_Train_Test(batch_size, img_size, test_input_img_paths, test_target_img_paths,num_classes)


#========================================================================
# ================== MODEL=========================================
#========================================================================

def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int): #, patch_size: int
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    # x = layers.Conv2D(filters, kernel_size=(image_x,image_y) )(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def get_conv_mixer_256_8(
     filters=64, depth=8, kernel_size=3, patch_size=2,  num_classes=num_classes, model_img_size= (image_x,image_y,num_channels) ): #depth=8, kernel_size=5  patch_size=2,
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = tf.keras.Input(model_img_size)
    # x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(inputs, filters, patch_size) #, patch_size

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x)
    # x = layers.GlobalAvgPool2D()(x)
    x= layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    outputs = layers.Conv2D(num_classes,(1,1), activation="sigmoid")(x) #softmax

    return Model(inputs, outputs)



#========================================================================
# =============== MODEL TRAINING AND EVALUATION =========================
#========================================================================


def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy", #sparse_ categorical
        metrics=["accuracy"],
    )

    checkpoint_filepath = os.path.abspath(base_dir+"/tmp/checkpoint")
    checkpoint_callback = ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, model


conv_mixer_model = get_conv_mixer_256_8()
history, conv_mixer_model = run_experiment(conv_mixer_model)
