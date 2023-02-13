
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Concatenate
from losses import my_softmax, my_kld, cgl_kl
import matplotlib.pyplot as plt
import os
import random


'''
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
'''
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
    
    
def gril_mse():
   
    mobilenet = tf.keras.applications.mobilenet.MobileNet(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(224,224,3),
    pooling=None,
    )

    mobilenet.trainable = True
    
   # RGB Channel
    rgb = Input(shape=(224,224,3), name='image')
       
    x = mobilenet(rgb, training = True)
    # x = L.BatchNormalization()(x)

    x = Conv2D(128, (5,5), strides=2, padding='same', activation='elu')(x)

    x = Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(x)
    
    x = Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(x)
 
    x = Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(x)
   
    x = Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(x)
  
    pool11 = MaxPool2D((2, 2), padding='same')(x)
    rgb_flat = Flatten()(pool11)
    rgb_flat = L.Dropout(0.5)(rgb_flat)

    # Depth Channel
    depth = Input(shape=(224,224,1), name='depth')
    # d = L.BatchNormalization()(depth)
    d = Conv2D(128, (5,5), strides=2, padding='same', activation='elu')(depth)

    d = Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(d)

    d = Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(d)

    d = Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(d)
 
    d = Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(d)

    pool21 = MaxPool2D((2, 2), padding='same')(d)

    depth_flat = Flatten()(pool21)
    depth_flat = L.Dropout(0.5)(depth_flat)

    shared_layer = tf.keras.layers.concatenate([rgb_flat, depth_flat])
 

    # action prediction head
    x1 = Dense(512, activation='elu')(shared_layer)
    x1 = L.Dropout(0.5)(x1)
    x1 = Dense(256, activation='elu')(x1)
    x1 = Dense(128, activation='elu')(x1)
    x1 = Dense(64, activation='elu')(x1)
    x1 = Dense(32, activation='elu')(x1)
    action = Dense(4, name='action')(x1)


    # gaze prediction head
    x2 = Dense(512, activation='relu')(shared_layer)
    x2 = L.Dropout(0.5)(x2) 
    x2 = Dense(256, activation='relu')(x2)
    x2 = Dense(128, activation='relu')(x2)
    x2 = Dense(64, activation='relu')(x2)
    gaze= Dense(2, name='gaze')(x2)


    model = Model(inputs = [rgb, depth], outputs=[action, gaze])
   
    model.summary() 
    return model


def agil_airsim(): 
    ###############################
    # Architecture of the network #
    ###############################
    num_action = 4 # act_roll, act_pitch, act_throttle, act_yaw
    SHAPE = (224,224, 1) # height * width * channel 
    dropout = 0.5

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


def vanilla_bc():

    inputs= Input(shape=(224, 224, 3), name="image")
    
    x=L.Conv2D(128, (5,5), strides=2, padding='same', activation='elu')(inputs)
    
    x=L.Conv2D(128, (5,5), strides=2, padding='same', activation='elu')(x)

    x=L.Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(x)
 
    x=L.Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(x)

    x=L.Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(x)

    x=L.Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(x)

    x= L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  
 
    x = L.Flatten()(x)
    x = L.Dropout(0.5)(x)

    x = L.Dense(256, activation='elu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(128, activation='elu')(x)
    #x = L.Dropout(0.5)(x)
    x = L.Dense(64, activation='elu')(x)

    output= Dense(4, name="action")(x)


    model=Model(inputs=inputs, outputs=output)

    return model
    

def vanilla_bc_depth():

    inputs= Input(shape=(224, 224, 1), name="image")
    
    x=L.Conv2D(128, (5,5), strides=2, padding='same', activation='elu')(inputs)
    
    x=L.Conv2D(128, (5,5), strides=2, padding='same', activation='elu')(x)

    x=L.Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(x)
 
    x=L.Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(x)

    x=L.Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(x)

    x=L.Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(x)

    x= L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  
 
    x = L.Flatten()(x)
    x = L.Dropout(0.5)(x)

    x = L.Dense(256, activation='elu')(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(128, activation='elu')(x)
    #x = L.Dropout(0.5)(x)
    x = L.Dense(64, activation='elu')(x)

    output= Dense(4, name="action")(x)


    model=Model(inputs=inputs, outputs=output)

    return model

def il_cgl_4():

    inputs= Input(shape=(224, 224, 3), name="image")

    x=L.Conv2D(128, (5,5), strides=2, padding='same', activation='elu')(inputs)

    x=L.Conv2D(128, (5,5), strides=2, padding='same', activation='elu')(x)

    x=L.Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(x)

    x=L.Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(x)

    #x=L.Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(x)

    #x=L.Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(x)

    # CGL conv output
    last_conv = L.Conv2D(1, (1,1), strides=1, padding='same')
    z = last_conv(x)
    cgl_out = L.Activation(my_softmax, name="gaze")(z)

    y = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    y = L.Flatten()(y)
    y = L.Dropout(0.5)(y)

    y = L.Dense(256, activation='elu')(y)
    y = L.Dropout(0.5)(y)
    y = L.Dense(128, activation='elu')(y)
    #x = L.Dropout(0.5)(x)
    y = L.Dense(64, activation='elu')(y)

    action = Dense(4, name="action")(y)


    model = Model(inputs=inputs, outputs=[cgl_out, action])

    return model


def il_cgl_3():

    inputs= Input(shape=(224, 224, 3), name="image")

    x=L.Conv2D(128, (5,5), strides=2, padding='same', activation='elu')(inputs)

    x=L.Conv2D(128, (5,5), strides=2, padding='same', activation='elu')(x)

    x=L.Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(x)

    # x=L.Conv2D(64, (5,5), strides=2, padding='same', activation='elu')(x)

    # x=L.Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(x)

    # x=L.Conv2D(32, (5,5), strides=2, padding='same', activation='elu')(x)

    # CGL conv output
    last_conv = L.Conv2D(1, (1,1), strides=1, padding='same')
    z = last_conv(x)
    cgl_out = L.Activation(my_softmax, name="gaze")(z)

    y = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    y = L.Flatten()(y)
    y = L.Dropout(0.5)(y)

    y = L.Dense(256, activation='elu')(y)
    y = L.Dropout(0.5)(y)
    y = L.Dense(128, activation='elu')(y)
    #x = L.Dropout(0.5)(x)
    y = L.Dense(64, activation='elu')(y)

    action = Dense(4, name="action")(y)


    model = Model(inputs=inputs, outputs=[cgl_out, action])

    return model
