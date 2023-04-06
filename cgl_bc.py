'''Example network architecture for behavioral cloning + CGL'''
import tensorflow as tf, numpy as np, tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model, Sequential 

def cgl_kl(y_true, y_pred):
    '''CGL loss function'''
    epsilon = 2.2204e-16 # introduce epsilon to avoid log and division by zero error
    y_true2 = K.backend.clip(y_true, epsilon, 1)
    y_pred = K.backend.clip(y_pred, epsilon, 1)
    return K.backend.sum(y_true * K.backend.log(y_true2 / y_pred)) #for old Keras need axis = [1,2,3]

def my_softmax(x):
    '''Softmax activation function. Normalize the whole metrics.
    # Arguments
        x : Tensor.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    '''
    return K.activations.softmax(x, axis=[1,2,3])

BATCH_SIZE = 50 
num_epoch = 50
num_action = 18
dropout = 0.0
gaze_weight = 0.01 #the alpha value in the loss function, other options are 0.05, 0.1, 0.3, 0.5, 0.7, 0.9 
SHAPE = (84,84,1) # height * width * channel
heatmap_shape = 21 # shape of the convolutional layer output feature map we want to supervise

if True:
    ###############################
    # Architecture of the network #
    ###############################

    inputs=L.Input(shape=SHAPE)
    x=inputs # inputs is used by the line "Model(inputs, ... )" below
    
    conv1=L.Conv2D(32, (8,8), strides=2, padding='same')
    x = conv1(x)
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)
    
    conv2=L.Conv2D(64, (4,4), strides=2, padding='same')
    x = conv2(x)
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)
    
    conv3=L.Conv2D(64, (3,3), strides=1, padding='same')
    x = conv3(x)
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)
    
    y=L.Flatten()(x)
    y=L.Dense(512, activation='relu')(y)
    y=L.Dropout(dropout)(y)
    logits=L.Dense(num_action, name="logits")(y)
    prob=L.Activation('softmax', name="prob")(logits)
   
    #GCL otuput
    conv4 = L.Conv2D(1, (1,1), strides=1, padding='same')
    z = conv4(x)
    conv_output = L.Activation(my_softmax, name="gaze_cvg")(z)

    model=Model(inputs=inputs, outputs=[conv_output, logits, prob])
    model.summary()
    model.count_params()

    opt=K.optimizers.Adadelta()

    model.compile(optimizer=opt, \
    loss={"gaze_cvg": cgl_kl, "logits": None, "prob": K.losses.sparse_categorical_crossentropy},\
    loss_weights={"prob":1 - gaze_weight, "gaze_cvg": gaze_weight})

if __name__ == "__main__":
    # LOAD the Atari-HEAD Dataset in your way
    from load_data import *
    d = Dataset(sys.argv[1], sys.argv[2]) #tarfile (images), txtfile (labels)
    d.standardize() 
    d.load_predicted_gaze_heatmap(sys.argv[3]) #npz file (predicted gaze heatmap)
    d.reshape_heatmap_for_cgl(heatmap_shape)

    model.fit(d.train_imgs, {"prob": d.train_lbl, "gaze_cvg": d.train_GHmap}, BATCH_SIZE, epochs=num_epoch,
        shuffle=True, verbose=2)

    model.save("model.hdf5")
