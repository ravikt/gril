import tensorflow as tf
import numpy as np
import os
import random
from models import agil_gaze_model, my_kld, my_softmax
from batch_loader import generate_gaze




batch_size = 4
csv_logger = tf.keras.callbacks.CSVLogger('gaze_adam.log')
datapath = "/scratch/user/ravikt/airsim/truck_mount_train/"
file_list = os.listdir(datapath)

random.shuffle(file_list)

# The file_list here should be the lenght of elements in npz files!
steps_per_epoch = np.int(np.ceil(len(file_list)/batch_size))
#x = generate_gaze(datapath, file_list)


tfx = tf.data.Dataset.from_generator(generate_gaze,args=[datapath, file_list], output_types = (tf.float32, tf.float32))
tfx = tfx.batch(batch_size)
model = agil_gaze_model()
#opt = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
opt=tf.keras.optimizers.Adam(learning_rate=0.0003)
model.compile(loss=my_kld, optimizer=opt)

model.fit(tfx, epochs=100, callbacks = [csv_logger])

#model.fit(tfx, steps_per_epoch=938, epochs=50, callbacks = [csv_logger])
model.save('gaze_adam_4.h5')

