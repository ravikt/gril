'''Example network architecture for AGIL (Attention-guided imitation learning'''
import tensorflow as tf, numpy as np, tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model, Sequential 

BATCH_SIZE = 50
num_epoch = 50
num_action = 18
SHAPE = (84,84,1) # height * width * channel 
dropout = 0.0

if True: 
    ###############################
    # Architecture of the network #
    ###############################

    gaze_heatmaps = L.Input(shape=(SHAPE[0],SHAPE[1],1))
    g=gaze_heatmaps
    g=L.BatchNormalization()(g)

    imgs=L.Input(shape=SHAPE)
    x=imgs
    x=L.Multiply()([x,g])
    x_intermediate=x
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
    # ============================ channel 2 ============================
    orig_x=imgs
    orig_x=L.Conv2D(32, (8,8), strides=2, padding='same')(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)
    orig_x=L.Dropout(dropout)(orig_x)

    orig_x=L.Conv2D(64, (4,4), strides=2, padding='same')(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)
    orig_x=L.Dropout(dropout)(orig_x)

    orig_x=L.Conv2D(64, (3,3), strides=1, padding='same')(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)

    x=L.Average()([x,orig_x])
    x=L.Dropout(dropout)(x)
    x=L.Flatten()(x)
    x=L.Dense(512, activation='relu')(x)
    x=L.Dropout(dropout)(x)
    logits=L.Dense(num_action, name="logits")(x)
    prob=L.Activation('softmax', name="prob")(logits)

    model=Model(inputs=[imgs, gaze_heatmaps], outputs=[logits, prob, g, x_intermediate])
    model.summary()

    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model.compile(loss={"prob":K.losses.sparse_categorical_crossentropy, "logits": None},
                 optimizer=opt)

if __name__ == "__main__":
    # LOAD the Atari-HEAD Dataset in your way
    from load_data import *
    d = Dataset(sys.argv[1], sys.argv[2]) #tarfile (images), txtfile (labels)
    d.load_predicted_gaze_heatmap(sys.argv[3]) #npz file (predicted gaze heatmap)
    d.standardize() 

    model.fit([d.train_imgs, d.train_GHmap], d.train_lbl, BATCH_SIZE, epochs=num_epoch, shuffle=True,verbose=2)
    model.save("model.hdf5")



