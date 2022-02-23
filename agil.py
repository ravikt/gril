'''Network architecture for AGIL (Attention-guided imitation learning'''
'''for AirSim data. Implmentation done as a part of cycle-of-learning project'''
import logging
#import pydotplus
import tensorflow as tf 
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras.utils import plot_model
#from tensforflow.keras.models import Model, Sequential
#from load_agil_data import some_loader

BATCH_SIZE = 2#50
num_epoch = 50
num_action = 4 # act_roll, act_pitch, act_throttle, act_yaw
SHAPE = (224,224,1) # height * width * channel 
dropout = 0.5

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

    x=L.Conv2D(128, (5,5), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)

    #x=L.Conv2D(128, (5,5), strides=2, padding='same')(x)
    #x=L.BatchNormalization()(x)
    #x=L.Activation('relu')(x)
 
    x=L.Conv2D(64, (5,5), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)

    x=L.Conv2D(64, (5,5), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    #x=L.Dropout(dropout)(x)

    x=L.Conv2D(32, (5,5), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    #x=L.Dropout(dropout)(x)

    x=L.Conv2D(32, (5,5), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)

    
    #x=L.Conv2D(32, (5,5), strides=2, padding='same')(x)
    #x=L.BatchNormalization()(x) 
    #x=L.Activation('relu')(x)
   
    x=L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x=L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # ============================ channel 2 ============================
    orig_x=imgs

    orig_x=L.Conv2D(128, (5,5), strides=2, padding='same')(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)

    
    #orig_x=L.Conv2D(128, (5,5), strides=2, padding='same')(orig_x)
    #orig_x=L.BatchNormalization()(orig_x)
    #orig_x=L.Activation('relu')(orig_x)


    orig_x=L.Conv2D(64, (5,5), strides=2, padding='same')(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)

    orig_x=L.Conv2D(64, (5,5), strides=2, padding='same')(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)
    #orig_x=L.Dropout(dropout)(orig_x)

    orig_x=L.Conv2D(32, (5,5), strides=2, padding='same')(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)
    #orig_x=L.Dropout(dropout)(orig_x)

    orig_x=L.Conv2D(32, (5,5), strides=2, padding='same')(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)

    #orig_x=L.Conv2D(32, (5,5), strides=2, padding='same')(orig_x)
    #orig_x=L.BatchNormalization()(orig_x)
    #orig_x=L.Activation('relu')(orig_x)
    orig_x= L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(orig_x)
    orig_x= L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(orig_x)
  
 
    x=L.Average()([x,orig_x])
    #x=L.Dropout(dropout)(x)
    x=L.Flatten()(x)
    x=L.Dropout(dropout)(x)
    x=L.Dense(512, activation='elu')(x)
    x=L.Dense(256, activation='elu')(x)
    x=L.Dense(128, activation='elu')(x)
    #x=L.Dropout(dropout)(x)
    output=L.Dense(num_action, name='action', activation='elu')(x)    

    #logits=L.Dense(num_action, name="logits")(x)
    #prob=L.Activation('softmax', name="prob")(logits)
    #prob = tf.nn.softmax(logits)
    #model=keras.Model(inputs=[imgs, gaze_heatmaps], outputs=[logits, prob, g, x_intermediate])
    
    model=keras.Model(inputs=[imgs, gaze_heatmaps], outputs=output)
    model.summary()

    opt=keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
   
    #model.compile(loss={"prob":None, "logits": keras.losses.SparseCategoricalCrossentropy()},
    #            optimizer=opt,metrics=[keras.metrics.SparseCategoricalAccuracy()])

    model.compile(loss='mean_squared_error', optimizer=opt)
    #plot_model(model)    

if __name__ == "__main__":
    # Load the Airsim Data
    import argparse
    parser = argparse.ArgumentParser(description="AGIL Network Architecture")
    
    parser.add_argument("-d", "--data", type=str, help="path to dataset" )
    #parser.add_argument("-l", "--labels", type=str, help="path to action labels")
    #parser.add_argument("-g", "--ghmap",  type=str, help="path to predicted gaze heatmap")
    #d = Dataset(sys.argv[1], sys.argv[2]) #tarfile (images), txtfile (labels)
    #d.load_predicted_gaze_heatmap(sys.argv[3]) #npz file (predicted gaze heatmap)
    #d.standardize() 
    
    args = parser.parse_args()
    data_path = args.data
    #lbls_path = args.labels
    
    with np.load(data_path) as data:
        l = len(data["images"])
        train_imgs = data['images']
        train_imgs = np.reshape(data['images'], (l, 224, 224, 1))
        print(train_imgs.shape)
        train_gaze = data['heatmap']
        train_gaze = np.reshape(data['heatmap'], (l, 224, 224, 1)) 
        print(train_gaze.shape)
        train_act = data['vel_comm']
        print(train_act.shape)

    model.fit([train_imgs, train_gaze], train_act, BATCH_SIZE, epochs=num_epoch, shuffle=True)
    model.save("agil_model.h5")
