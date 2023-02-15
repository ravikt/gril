import numpy as np
import tensorflow as tf
from tensorflow import keras
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

def action_loss(y_true, y_pred):
    """
    Weighted MSE for control commands. This loss function gives
    higher weightage to errors in roll and yaw commands
    The weight coefficient are [0.40, 0.10,0.10, 0.40]
    for roll, pitch, throttle and yaw respectively
    """

    squared_difference = tf.square(y_true - y_pred)
    weights = np.array([[0.005, 0.005, 0.005, 0.85]])
    #weights = np.array([[0.40, 0.10, 0.10, 0.40]])
    weights = np.array([[0.10, 0.10, 0.10, 0.70]])
    weighted_squared_difference = weights*squared_difference

    return tf.reduce_mean(weighted_squared_difference, axis=-1)