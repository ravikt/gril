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

def gril():
   
    mobilenet = tf.keras.applications.mobilenet.MobileNet(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(224,224,3),
    pooling=None,
    )

    mobilenet.trainable = False
    
   # RGB Channel
    rgb = Input(shape=(224,224,3), name='image')
    
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
    

def agil_airsim(): 
    ###############################
    # Zhang et.al "AGIL: Learning Attention from Human for Visuomotor Tasks"
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
    

def il_cgl():
    
    
  # RGB Channel
    rgb = Input(shape=(224,224,3), name='image')
       

    # inputs= Input(shape=(224, 224, 3), name="image")

    x=L.Conv2D(128, (5,5), strides=2, padding='same', activation='elu')(rgb)

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


    model = Model(inputs=rgb, outputs=[cgl_out, action])

    return model
    
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

