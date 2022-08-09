'''This file reads a trained gaze prediction network by Zhang et al. 2020, and a data file, then outputs human attention map
Zhang, R., Walshe, C., Liu, Z., Guan, L., Muller, K., Whritner, J., ... & Ballard, D. (2020, April). Atari-head: Atari human eye-tracking and demonstration dataset. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 04, pp. 6811-6820).'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
import matplotlib.pyplot as plt
import os
import random


def my_softmax(x):
    """Softmax activation function. Normalize the whole metrics.
    # Arguments
        x : Tensor.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    return keras.activations.softmax(x, axis=[1, 2, 3])


def my_kld(y_true, y_pred):
    """
    Correct keras bug. Compute the KL-divergence between two metrics.
    """
    epsilon = 1e-10  # introduce epsilon to avoid log and division by zero error
    y_true = keras.backend.clip(y_true, epsilon, 1)
    y_pred = keras.backend.clip(y_pred, epsilon, 1)
    return keras.backend.sum(y_true * keras.backend.log(y_true / y_pred), axis=[1, 2, 3])

def agil_gaze_model():
    # Constants
    img_shape = 224 #84
    k = 1
    # Constants
    SHAPE = (img_shape, img_shape, 1)  # height * width
    dropout = 0.0
    ###############################
    # Architecture of the network #
    ###############################
    imgs  = L.Input(shape=SHAPE) 
    #x     = L.Reshape((224,224,1))(imgs)
    x = imgs

    conv1 = L.Conv2D(64, (5, 5), strides=2, padding='same')
    x = conv1(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)
    
    conv2 = L.Conv2D(32, (3, 3), strides=2, padding='same')
    x = conv2(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)

    conv3 = L.Conv2D(32, (4, 4), strides=2, padding='same')
    x = conv3(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)

    conv4 = L.Conv2D(64, (3, 3), strides=2, padding='same')
    x = conv4(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)

    deconv1 = L.Conv2DTranspose(64, (3, 3), strides=2, padding='same')
    x = deconv1(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)

    deconv2 = L.Conv2DTranspose(32, (3, 3), strides=2, padding='same')
    x = deconv2(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)
    
    deconv3 = L.Conv2DTranspose(32, (3, 3), strides=2, padding='same')
    x = deconv3(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)


    deconv4 = L.Conv2DTranspose(1, (5, 5), strides=2, padding='same')
    x = deconv4(x)

    outputs = L.Activation(my_softmax)(x)
    model = keras.Model(inputs=imgs, outputs=outputs)

    model.summary()

    return model
    

def gaze_coord_model():
    img_shape = 224 #84
    k = 1
    # Constants
    SHAPE = (img_shape, img_shape, 1)  # height * width
    dropout = 0.0
    ###############################
    # Architecture of the network #
    ###############################
    imgs  = L.Input(shape=SHAPE)
    #x     = L.Reshape((224,224,1))(imgs)
    x = imgs

    conv1 = L.Conv2D(64, (5, 5), strides=2, padding='same')
    x = conv1(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)

    conv2 = L.Conv2D(32, (3, 3), strides=2, padding='same')
    x = conv2(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)
    
    conv3 = L.Conv2D(32, (4, 4), strides=2, padding='same')
    x = conv3(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)

    conv4 = L.Conv2D(64, (3, 3), strides=2, padding='same')
    x = conv4(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)
    
    x = L.Flatten()(x)
    x = L.Dropout(0.4)(x)
    
    x = L.Dense(64, activation='relu')(x)
    x = L.Dense(32, activation='relu')(x)
    x = L.Dense(16, activation='relu')(x)
    #x = Dense(64, activation='tanh')(x)
    # Output probability
    # Sigmoid tells us the probability if input sceneflow 
    # is real or not
    outputs = L.Dense(2, activation='sigmoid')(x)
    model = keras.Model(inputs=imgs, outputs=outputs)

    model.summary()

    return model

# def agil_gaze_model():
#     # Constants
#     img_shape = 224 #84
#     k = 1
#     # Constants
#     SHAPE = (img_shape, img_shape, 1)  # height * width
#     dropout = 0.0
#     ###############################
#     # Architecture of the network #
#     ###############################
#     imgs  = L.Input(shape=SHAPE) 
#     #x     = L.Reshape((224,224,1))(imgs)
#     x = imgs

#     conv1 = L.Conv2D(32, (8, 8), strides=4, padding='same')
#     x = conv1(x)
#     x = L.Activation('relu')(x)
#     x = L.BatchNormalization()(x)
#     x = L.Dropout(dropout)(x)

#     conv2 = L.Conv2D(64, (4, 4), strides=2, padding='same')
#     x = conv2(x)
#     x = L.Activation('relu')(x)
#     x = L.BatchNormalization()(x)
#     x = L.Dropout(dropout)(x)

#     conv3 = L.Conv2D(64, (3, 3), strides=1, padding='same')
#     x = conv3(x)
#     x = L.Activation('relu')(x)
#     x = L.BatchNormalization()(x)
#     x = L.Dropout(dropout)(x)

#     deconv1 = L.Conv2DTranspose(64, (3, 3), strides=1, padding='same')
#     x = deconv1(x)
#     x = L.Activation('relu')(x)
#     x = L.BatchNormalization()(x)
#     x = L.Dropout(dropout)(x)

#     deconv2 = L.Conv2DTranspose(32, (4, 4), strides=2, padding='same')
#     x = deconv2(x)
#     x = L.Activation('relu')(x)
#     x = L.BatchNormalization()(x)
#     x = L.Dropout(dropout)(x)

#     deconv3 = L.Conv2DTranspose(1, (8, 8), strides=4, padding='same')
#     x = deconv3(x)

#     outputs = L.Activation(my_softmax)(x)
#     model = keras.Model(inputs=imgs, outputs=outputs)

#     model.summary()

#     return model


#BATCH_SIZE = 1#50
#num_epoch = 50
num_action = 4 # act_roll, act_pitch, act_throttle, act_yaw
SHAPE = (224,224, 1) # height * width * channel 
dropout = 0.5

def agil_airsim(): 
    ###############################
    # Architecture of the network #
    ###############################

    gaze_heatmaps = L.Input(shape=(SHAPE), name='gaze')
    g=L.BatchNormalization()(gaze_heatmaps)

    imgs=L.Input(shape=SHAPE, name='images')
    #x=L.Reshape((224, 224, 1))(imgs)
    x = imgs
    x = L.Multiply()([x,g])
    x_intermediate=x

    x=L.Conv2D(128, (5,5), strides=2, padding='same', activation='elu')(x)
 
    x=L.Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(x)

    x=L.Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(x)
    #x=L.Dropout(dropout)(x)

    x=L.Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(x)
    #x=L.Dropout(dropout)(x)

    x=L.Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(x)
   
    x=L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x=L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # ============================ channel 2 ============================
    orig_x=imgs
   
    orig_x=L.Conv2D(128, (5,5), strides=2, padding='same', activation='elu')(orig_x)

    orig_x=L.Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(orig_x)
 
    orig_x=L.Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(orig_x)

    orig_x=L.Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(orig_x)

    orig_x=L.Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(orig_x)

    #orig_x=L.Conv2D(32, (5,5), strides=2, padding='same')(orig_x)
    #orig_x=L.BatchNormalization()(orig_x)
    #orig_x=L.Activation('relu')(orig_x)
    orig_x= L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(orig_x)
    orig_x= L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(orig_x)
  
 
    x=L.Average()([x,orig_x])
    #x=L.Dropout(dropout)(x)
    x=L.Flatten()(x)
    x=L.Dropout(dropout)(x)
    # ReLU??
    x=L.Dense(512, activation='elu')(x)
    x=L.Dense(256, activation='elu')(x)
    x=L.Dense(128, activation='elu')(x)
    #x=L.Dropout(dropout)(x)
    output=L.Dense(num_action, name='action')(x)    

    agil_airsim_model=keras.Model(inputs=[imgs, gaze_heatmaps], outputs=output)
    agil_airsim_model.summary()
    
    return agil_airsim_model
