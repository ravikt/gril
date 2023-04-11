import tensorflow as tf
import numpy as np
import os
import random
from models import gril, my_kld, my_softmax
from batch_loader import generate_aril


def action_loss(y_true, y_pred):
    """
    Weighted MSE for control commands. This loss function gives
    higher weightage to errors in roll and yaw commands
    The weight coefficient are [0.40, 0.10,0.10, 0.40]
    for roll, pitch, throttle and yaw respectively
    """

    squared_difference = tf.square(y_true - y_pred)
    #weights = np.array([[0.40, 0.10, 0.10, 0.40]])
    # weights = np.array([[0.005, 0.005, 0.005, 0.85]])
    weights = np.array([[0.10, 0.10, 0.10, 0.70]])

    weighted_squared_difference = weights*squared_difference

    return tf.reduce_mean(weighted_squared_difference, axis=-1)

#64
batch_size = 32
train_datapath = "/scratch/user/ravikt/airsim/data/dataset_cvpr/raw_flipped/train/"
val_datapath =  "/scratch/user/ravikt/airsim/data/dataset_cvpr/raw_flipped/val/"
 
 

file_list = os.listdir(train_datapath)
val_list = os.listdir(val_datapath)

random.shuffle(file_list)

# The file_list here should be the lenght of elements in npz files!
steps_per_epoch = np.int(np.ceil(len(file_list)/batch_size))
val_steps = np.int(np.ceil(len(val_list)/batch_size))


my_callbacks = [
    tf.keras.callbacks.CSVLogger('gil.log')
]

    
tfx = tf.data.Dataset.from_generator(generate_aril, args=[train_datapath, file_list], 
                                     output_types = ({"image":tf.float32, 
                                     "depth":tf.float32}, {"action":tf.float64, "gaze":tf.float64}))

val = tf.data.Dataset.from_generator(generate_aril, args=[val_datapath, val_list], 
                                     output_types = ({"image":tf.float32, 
                                     "depth":tf.float32}, {"action":tf.float64, "gaze":tf.float64}))

 


tfx = tfx.batch(batch_size)
val = val.batch(batch_size)
model = gril()


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.00001,
    decay_steps=10000,
    decay_rate=0.9)

opt=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=10e-4)
model.compile(loss=[action_loss, 'mean_squared_error'], optimizer=opt)

model.fit(tfx, epochs=30, validation_data=val, validation_steps=val_steps, callbacks = my_callbacks)

model.save('gil.h5')

