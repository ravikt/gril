'''Example network architecture for behavioral cloning'''
import tensorflow as tf, numpy as np
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model, Sequential 

BATCH_SIZE = 50
num_epoch = 50
num_action = 18
SHAPE = (84,84,1) # height * width * channel
dropout = 0.0

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

    opt=K.optimizers.legacy.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(loss={"prob":K.losses.sparse_categorical_crossentropy, "logits": None},
                 optimizer=opt)

if __name__ == "__main__":
    # LOAD the Atari-HEAD Dataset in your way
    from load_data import *
    import glob
    # d = Dataset(sys.argv[1], sys.argv[2]) #tarfile (images), txtfile (labels)
    # d.standardize()

    file = "/home/grads/m/mdsunbeam/atari_head/IL-CGL/data/alien"
    i = 0
    action_arr = []
    # gaze_arr = []
    img_arr = []
    for f in glob.glob(file + "/*.txt"):
        print(f)
        d = Dataset(f[0:-3] + "tar.bz2", f) #tarfile (images), txtfile (labels)
        # For gaze prediction
        d.generate_data_for_gaze_prediction()
        d.standardize() #for training imitation learning only, gaze model has its own mean files

        action_arr.append(d.action_train_lbl)
        # gaze_arr.append(d.train_gaze)
        img_arr.append(d.train_imgs)
        i += 1
        # if i == 2:
        #     break

    action_arr = np.concatenate(action_arr)
    action_arr = np.vstack(action_arr)
    # gaze_arr = np.concatenate(gaze_arr)
    # gaze_arr = np.vstack(gaze_arr)
    # img_arr = np.concatenate(img_arr)
    img_arr = np.vstack(img_arr)


    # action_arr = np.asarray(action_arr, dtype=np.int64)
    # gaze_arr = np.asarray(gaze_arr, dtype=np.int64)
    # img_arr = np.asarray(img_arr, dtype=np.float32)

    print(action_arr.shape)
    # print(gaze_arr.shape)
    print(img_arr.shape)
    
    # print(action_arr[66])
    # print(gaze_arr[66])
    # print(img_arr[66])
    model.fit(img_arr, action_arr, BATCH_SIZE, epochs=num_epoch, shuffle=True, verbose=2)
    model.save("bc_alien.h5")

    # model.fit(d.train_imgs, d.action_train_lbl, BATCH_SIZE, epochs=num_epoch, shuffle=True, verbose=2)
    # model.save("model.hdf5")
