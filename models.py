"""Model architectures for gaze-regularized imitation learning.

This module implements multiple model architectures:
- GRIL: Gaze Regularized Imitation Learning (dual-head RGB+Depth)
- AGIL: Attention-Guided Imitation Learning
- IL-CGL: Imitation Learning with Controllable Gaze Localization
- Vanilla BC: Vanilla Behavioral Cloning baseline
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D
from losses import my_softmax

# Architecture constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS_RGB = 3
IMG_CHANNELS_GRAY = 1
NUM_ACTIONS = 4  # roll, pitch, throttle, yaw
NUM_GAZE_COORDS = 2  # x, y coordinates

# Network hyperparameters
DENSE_UNITS = [512, 256, 128, 64]
CONV_FILTERS_RGB = [64, 64]
CONV_FILTERS_DEPTH = [64, 64, 32, 16]
DROPOUT_RATE = 0.5


def gril():
    """Build GRIL (Gaze Regularized Imitation Learning) model.
    
    Multi-input, multi-output architecture that processes RGB and depth images
    to predict both control actions and gaze coordinates. Uses MobileNet as 
    feature extractor for RGB channel.
    
    Returns:
        keras.Model: Model with inputs ['image', 'depth'] and outputs ['action', 'gaze']
    """
    # MobileNet feature extractor for RGB channel
    mobilenet = tf.keras.applications.MobileNet(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_RGB),
        pooling=None,
    )
    mobilenet.trainable = False
    
    # RGB Channel processing
    rgb_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_RGB), name='image')
    rgb_features = mobilenet(rgb_input, training=False)
    rgb_features = Conv2D(CONV_FILTERS_RGB[0], (5, 5), strides=2, padding='same', activation='relu')(rgb_features)
    rgb_features = Conv2D(CONV_FILTERS_RGB[1], (5, 5), strides=2, padding='same', activation='relu')(rgb_features)
    rgb_features = MaxPool2D(pool_size=(2, 2))(rgb_features)
    rgb_flat = Flatten()(rgb_features)

    # Depth Channel processing
    depth_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_RGB), name='depth')
    depth_features = Conv2D(CONV_FILTERS_DEPTH[0], (5, 5), strides=2, padding='same', activation='relu')(depth_input)
    depth_features = Conv2D(CONV_FILTERS_DEPTH[1], (5, 5), strides=2, padding='same', activation='relu')(depth_features)
    depth_features = Conv2D(CONV_FILTERS_DEPTH[2], (5, 5), strides=2, padding='same', activation='relu')(depth_features)
    depth_features = Conv2D(CONV_FILTERS_DEPTH[3], (5, 5), strides=2, padding='same', activation='relu')(depth_features)
    depth_features = MaxPool2D(pool_size=(2, 2))(depth_features)
    depth_flat = Flatten()(depth_features)
    
    # Merge RGB and Depth features
    shared_features = tf.keras.layers.concatenate([rgb_flat, depth_flat])

    # Action prediction head
    action_head = Dense(DENSE_UNITS[0], activation='elu')(shared_features)
    action_head = Dense(DENSE_UNITS[1], activation='elu')(action_head)
    action_head = Dense(DENSE_UNITS[2], activation='elu')(action_head)
    action_head = Dense(DENSE_UNITS[3], activation='elu')(action_head)
    action_output = Dense(NUM_ACTIONS, name='action')(action_head)

    # Gaze prediction head
    gaze_head = Dense(DENSE_UNITS[0], activation='relu')(shared_features)
    gaze_head = Dense(DENSE_UNITS[1], activation='relu')(gaze_head)
    gaze_head = Dense(DENSE_UNITS[2], activation='relu')(gaze_head)
    gaze_head = Dense(DENSE_UNITS[3], activation='relu')(gaze_head)
    gaze_output = Dense(NUM_GAZE_COORDS, name='gaze')(gaze_head)

    model = Model(inputs=[rgb_input, depth_input], outputs=[action_output, gaze_output])
   
    model.summary() 
    return model
    

def agil_airsim(): 
    """Build AGIL (Attention-Guided Imitation Learning) model for AirSim.
    
    Implementation of Zhang et al. "AGIL: Learning Attention from Human for Visuomotor Tasks".
    Uses gaze heatmaps to guide attention in a dual-stream architecture.
    
    Returns:
        keras.Model: Model with inputs ['images', 'gaze'] and output ['action']
    """
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_GRAY)
    
    # Gaze heatmap processing
    gaze_input = L.Input(shape=input_shape, name='gaze')
    gaze_normalized = L.BatchNormalization()(gaze_input)
    
    # Image input
    image_input = L.Input(shape=input_shape, name='images')
    
    # Attention-modulated stream (gaze-guided)
    attention_stream = L.Multiply()([image_input, gaze_normalized])
    attention_stream = L.Conv2D(128, (5, 5), strides=2, padding='same', activation='elu')(attention_stream)
    attention_stream = L.Conv2D(64, (5, 5), strides=2, padding='same', activation='elu')(attention_stream)
    attention_stream = L.Conv2D(64, (5, 5), strides=2, padding='same', activation='elu')(attention_stream)
    attention_stream = L.Conv2D(32, (5, 5), strides=2, padding='same', activation='elu')(attention_stream)
    attention_stream = L.Conv2D(32, (5, 5), strides=2, padding='same', activation='elu')(attention_stream)
    attention_stream = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(attention_stream)
    attention_stream = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(attention_stream)

    # Original stream (no attention)
    original_stream = L.Conv2D(128, (5, 5), strides=2, padding='same', activation='elu')(image_input)
    original_stream = L.Conv2D(64, (5, 5), strides=2, padding='same', activation='elu')(original_stream)
    original_stream = L.Conv2D(64, (5, 5), strides=2, padding='same', activation='elu')(original_stream)
    original_stream = L.Conv2D(32, (5, 5), strides=2, padding='same', activation='elu')(original_stream)
    original_stream = L.Conv2D(32, (5, 5), strides=2, padding='same', activation='elu')(original_stream)
    original_stream = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(original_stream)
    original_stream = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(original_stream)
    
    # Merge both streams
    merged_features = L.Average()([attention_stream, original_stream])
    merged_features = L.Flatten()(merged_features)
    merged_features = L.Dropout(DROPOUT_RATE)(merged_features)
    
    # Action prediction
    action_head = L.Dense(DENSE_UNITS[0], activation='elu')(merged_features)
    action_head = L.Dense(DENSE_UNITS[1], activation='elu')(action_head)
    action_head = L.Dense(DENSE_UNITS[2], activation='elu')(action_head)
    action_output = L.Dense(NUM_ACTIONS, name='action')(action_head)
    
    model = keras.Model(inputs=[image_input, gaze_input], outputs=action_output)
    model.summary()
    return model
    

def il_cgl():
    """Build IL-CGL (Imitation Learning with Controllable Gaze Localization) model.
    
    Multi-output architecture that jointly predicts gaze heatmaps and control actions.
    The gaze prediction uses a spatial softmax activation.
    
    Returns:
        keras.Model: Model with input 'image' and outputs ['gaze', 'action']
    """
    # Image input
    image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_RGB), name='image')
    
    # Shared convolutional features
    features = L.Conv2D(128, (5, 5), strides=2, padding='same', activation='elu')(image_input)
    features = L.Conv2D(128, (5, 5), strides=2, padding='same', activation='elu')(features)
    features = L.Conv2D(64, (5, 5), strides=2, padding='same', activation='elu')(features)
    
    # Gaze heatmap prediction branch
    gaze_conv = L.Conv2D(1, (1, 1), strides=1, padding='same')(features)
    gaze_output = L.Activation(my_softmax, name="gaze")(gaze_conv)
    
    # Action prediction branch
    action_features = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(features)
    action_features = L.Flatten()(action_features)
    action_features = L.Dropout(DROPOUT_RATE)(action_features)
    action_features = L.Dense(256, activation='elu')(action_features)
    action_features = L.Dropout(DROPOUT_RATE)(action_features)
    action_features = L.Dense(128, activation='elu')(action_features)
    action_features = L.Dense(64, activation='elu')(action_features)
    action_output = Dense(NUM_ACTIONS, name="action")(action_features)
    
    model = Model(inputs=image_input, outputs=[gaze_output, action_output])
    return model
    
def vanilla_bc():
    """Build Vanilla Behavioral Cloning baseline model.
    
    Simple CNN architecture without gaze information. Used as baseline
    for comparison with gaze-augmented methods.
    
    Returns:
        keras.Model: Model with input 'image' and output 'action'
    """
    image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_RGB), name="image")
    
    # Convolutional feature extraction
    features = L.Conv2D(128, (5, 5), strides=2, padding='same', activation='elu')(image_input)
    features = L.Conv2D(128, (5, 5), strides=2, padding='same', activation='elu')(features)
    features = L.Conv2D(64, (5, 5), strides=2, padding='same', activation='elu')(features)
    features = L.Conv2D(64, (5, 5), strides=2, padding='same', activation='elu')(features)
    features = L.Conv2D(32, (5, 5), strides=2, padding='same', activation='elu')(features)
    features = L.Conv2D(32, (5, 5), strides=2, padding='same', activation='elu')(features)
    features = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(features)
    
    # Action prediction
    features = L.Flatten()(features)
    features = L.Dropout(DROPOUT_RATE)(features)
    features = L.Dense(256, activation='elu')(features)
    features = L.Dropout(DROPOUT_RATE)(features)
    features = L.Dense(128, activation='elu')(features)
    features = L.Dense(64, activation='elu')(features)
    action_output = Dense(NUM_ACTIONS, name="action")(features)
    
    model = Model(inputs=image_input, outputs=action_output)
    return model

