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

import mat73
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"



print("Packages Loaded")

#========================================================================
# ==================LOAD DATA =========================================
#========================================================================

# Might need to revise this: this might not be the best approach considering memory

base_dir = r'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Full_block_len_45_280921_1530'

raw_data = mat73.loadmat(os.path.abspath(base_dir+'/zero_class_reduced_echo_cnn_45x15.mat')) 

all_data = raw_data['echo_cnn_input']
all_target = raw_data['echo_cnn_target']

num_classes = int(max(np.unique(all_target)))
all_target[all_target == num_classes ] = 0

# Train-test split
shuffle = 0
if shuffle:
    random.Random(1337).shuffle(all_data)
    random.Random(1337).shuffle(all_target)
    

## Prep data
train_size = int(np.floor(0.85*len(all_target)));
test_size = int(np.round( 0.1* all_data.shape[0] ))
val_size = all_data.shape[0] -train_size - test_size


x_train = all_data[0:train_size,:]
x_train = np.reshape( x_train, (x_train.shape[0],num_classes-1,-1) )

x_test = all_data[train_size:train_size+test_size,:]
x_test = np.reshape( x_test,(x_test.shape[0],num_classes-1,-1) )

x_val = all_data[-val_size:,:]
x_val = np.reshape( x_val,(x_val.shape[0],num_classes-1,-1) )

y_train = all_target[:train_size]
y_test  = all_target[train_size:train_size+test_size]
y_val = all_target[-val_size:]


# Convert labels to categorical orthonormal vectors (One-hot encodiing)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test, num_classes)


print(f"Training data samples: {len(x_train)}")
print(f"Test data samples: {len(x_test)}")
print(f"Validation data samples: {len(x_val)}")


# AUGMENTATION
image_x,image_y = x_train.shape[1:]
img_size = (image_x,image_y)

auto = tf.data.AUTOTUNE

data_augmentation = Sequential(
    [layers.RandomCrop(image_x, image_y), layers.RandomFlip("vertical"),],
    name="data_augmentation",
)


# Hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32 #128
num_epochs = 20
num_channels = 1


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

class Echo_Load_Train_Test(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, data, target, img_size = img_size, batch_size = batch_size, num_classes = num_classes):
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return len(self.target) // self.batch_size
    

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size    
        x = self.data[i : i + self.batch_size]
        x = np.expand_dims( x, -1)
        
        y = self.target[i : i + self.batch_size]
        y = np.expand_dims( y, -1)
            
        return x, y  

train_dataset = Echo_Load_Train_Test(x_train, y_train)
val_dataset = Echo_Load_Train_Test(x_val, y_val)
test_dataset = Echo_Load_Train_Test(x_test, y_test)


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
     filters=512, depth=16, kernel_size=3, patch_size=2,  num_classes=num_classes, image_size= (image_x,image_y,num_channels) ): #depth=8, kernel_size=5  patch_size=2,
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = tf.keras.Input((image_x,image_y,num_channels))
    # x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(inputs, filters, patch_size) #, patch_size

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

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
        loss="categorical_crossentropy", #sparse_
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
