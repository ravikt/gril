'''Example network architecture for behavioral cloning'''
import tensorflow as tf, numpy as np, keras as K
import keras.layers as L
from keras.models import Model, Sequential 

BATCH_SIZE = 50
num_epoch = 50
num_action = 18
SHAPE = (84,84,1) # height * width * channel
dropout =  0.0

if True: # I just want to indent
    ###############################
    # Architecture of the network #
    ###############################

    inputs=L.Input(shape=SHAPE)
    x=inputs # inputs is used by the line "Model(inputs, ... )" below
    x=L.Conv2D(32, (8,8), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    
    x=L.Conv2D(64, (4,4), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    
    x=L.Conv2D(64, (3,3), strides=1, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    x=L.Flatten()(x)
    
    x=L.Dense(512, activation='relu')(x)
    x=L.Dropout(dropout)(x)
    logits=L.Dense(num_action, name="logits")(x)
    prob=L.Activation('softmax', name="prob")(logits)
    model=Model(inputs=inputs, outputs=[logits, prob])
    model.summary()

    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(loss={"prob":None, "logits": K.metrics.sparse_categorical_accuracy},
                 optimizer=opt,metrics=[K.metrics.sparse_categorical_accuracy])

if __name__ == "__main__":
    # LOAD the Atari-HEAD Dataset in your way
    from load_data import *
    d = Dataset(sys.argv[1], sys.argv[2]) #tarfile (images), txtfile (labels)
    d.standardize()

    model.fit(d.train_imgs, d.train_lbl, BATCH_SIZE, epochs=num_epoch, shuffle=True, verbose=2)
    model.save("model.hdf5")
