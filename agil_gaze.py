'''This file reads a trained gaze prediction network by Zhang et al. 2020, and a data file, then outputs human attention map
Zhang, R., Walshe, C., Liu, Z., Guan, L., Muller, K., Whritner, J., ... & Ballard, D. (2020, April). Atari-head: Atari human eye-tracking and demonstration dataset. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 04, pp. 6811-6820).'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
import matplotlib.pyplot as plt
# from tensorflow.keras import models

# import tensorflow as tf, numpy as np, keras as K
# import keras.layers as L
# from keras.models import Model, Sequential


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
    SHAPE = (img_shape, img_shape, k)  # height * width * channel
    dropout = 0.0
    ###############################
    # Architecture of the network #
    ###############################
    inputs = L.Input(shape=SHAPE)
    x = inputs

    conv1 = L.Conv2D(32, (8, 8), strides=4, padding='same')
    x = conv1(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)

    conv2 = L.Conv2D(64, (4, 4), strides=2, padding='same')
    x = conv2(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)

    conv3 = L.Conv2D(64, (3, 3), strides=1, padding='same')
    x = conv3(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)

    deconv1 = L.Conv2DTranspose(64, (3, 3), strides=1, padding='same')
    x = deconv1(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)

    deconv2 = L.Conv2DTranspose(32, (4, 4), strides=2, padding='same')
    x = deconv2(x)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)

    deconv3 = L.Conv2DTranspose(1, (8, 8), strides=4, padding='same')
    x = deconv3(x)

    outputs = L.Activation(my_softmax)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    return model


# def predict_and_save(, imgs):
#     print "Predicting results..."
#     preds = model.predict(.imgs)
#     print "Predicted."

#     print "Writing predicted gaze heatmap (train) into the npz file..."
#     np.savez_compressed("human_gaze_" + .game_name, heatmap=.preds[:,:,:,0])
#     print "Done. Output is:"
#     print " %s" % "human_gaze_" + .game_name + '.npz'

if __name__ == "__main__":

    # shuffle_size=
    batch_size = 4
    path = "/scratch/user/ravikt/small.npz"
    with np.load(path) as data:
        l = len(data["images"])
        train_examples = np.reshape(data['images'], (l, 224, 224, 1))
        train_labels = np.reshape(data['heatmap'], (l, 224, 224, 1))

    # print(np.any(np.isnan(train_labels)))
    # print(tf.math.is_nan(train_labels[8]))
    model = agil_gaze_model()
    opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model.compile(loss=my_kld, optimizer=opt)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_examples, train_labels))
    # train_dataset = train_dataset.shuffle(shuffle_size).batch(batch_size)
    train_dataset = train_dataset.batch(batch_size)

    model.fit(train_dataset, epochs=50)
    model.save('small.h5')
