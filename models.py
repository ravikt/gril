# Models.py

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

def aril():
   
    resnet = tf.keras.applications.mobilenet.MobileNet(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(224,224,3),
    pooling=None,
    )

    resnet.trainable = False
    
   # RGB Channel
    rgb = Input(shape=(224,224,3), name='image')
    # conv11 = Conv2D(32, kernel_size=4, activation='relu')(rgb)
    # pool11 = MaxPool2D(pool_size=(2, 2))(conv11)
    # conv12 = Conv2D(16, kernel_size=4, activation='relu')(pool11)
    # pool12 = MaxPool2D(pool_size=(2, 2))(conv12)
    x = resnet(rgb, training = False)
  
    x = Conv2D(64, (5,5), strides=2, padding='same', activation='relu')(x)

    x = Conv2D(64, (5,5), strides=2, padding='same', activation='relu')(x)

    #conv13 = Conv2D(32, (5,5), strides=2, padding='same', activation='relu')(conv12)

    #conv14 = Conv2D(16, (5,5), strides=2, padding='same', activation='relu')(conv13)

 
    pool11 = MaxPool2D(pool_size=(2, 2))(x)
    rgb_flat = Flatten()(pool11)

    # Depth Channel
    depth = Input(shape=(224,224,3), name='depth')
    conv21 = Conv2D(64, (5,5), strides=2, padding='same', activation='relu')(depth)
  
    conv22 = Conv2D(64, (5,5), strides=2, padding='same', activation='relu')(conv21)

    conv23 = Conv2D(32, (5,5), strides=2, padding='same', activation='relu')(conv22)
  
    conv24 = Conv2D(16, (5,5), strides=2, padding='same', activation='relu')(conv23)
 
    pool21 = MaxPool2D(pool_size=(2, 2))(conv24)

    depth_flat = Flatten()(pool21)
    
    
    # Shared feature extraction layer
    # merge pool12 and pool22
    # shared_layer = Concatenate([pool12, pool22])

    shared_layer = tf.keras.layers.concatenate([rgb_flat, depth_flat])
 

    # action prediction head
    x1= Dense(512, activation='elu')(shared_layer)
    x1= Dense(256, activation='elu')(x1)
    x1= Dense(128, activation='elu')(x1)
    x1= Dense(64, activation='elu')(x1)
   
    # action = Dense(4, activation='softmax')(x1)
    action = Dense(4, name='action')(x1)


    # conv1 = Conv2D(32, kernel_size=3, activation='relu')(shared_layer)
    # pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    # action = Flatten()(pool1)

    # gaze prediction head
    x2= Dense(512, activation='relu')(shared_layer)
    x2= Dense(256, activation='relu')(x2)
    x2= Dense(128, activation='relu')(x2)
    x2= Dense(64, activation='relu')(x2)

    # gaze= Dense(2, activation='softmax')(x2)
    gaze= Dense(2, name='gaze')(x2)

    # conv2 = Conv2D(16, kernel_size=3, activation='relu')(shared_layer)
    # pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    # gaze = Flatten()(pool2)

    model = Model(inputs = [rgb, depth], outputs=[action, gaze])
   
    model.summary() 
    return model
