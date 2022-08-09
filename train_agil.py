import tensorflow as tf
import numpy as np
import os
import random
from models import agil_gaze_model, my_kld, my_softmax, agil_airsim
from batch_loader import generate

#64
batch_size = 32
datapath = "/scratch/user/ravikt/airsim/truck_mount_train/"
file_list = os.listdir(datapath)

random.shuffle(file_list)

# The file_list here should be the lenght of elements in npz files!
steps_per_epoch = np.int(np.ceil(len(file_list)/batch_size))

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2),
    tf.keras.callbacks.ModelCheckpoint('/scratch/user/ravikt/airsim/model-{epoch:03d}.h5', 
                                 monitor='loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto'),
    tf.keras.callbacks.CSVLogger('agil_sttr_32.log')
]

    
tfx = tf.data.Dataset.from_generator(generate,args=[datapath, file_list], 
                                     output_types = ({"images":tf.float32, 
                                     "gaze":tf.float32}, tf.float64))



tfx = tfx.batch(batch_size)
model = agil_airsim()

opt=tf.keras.optimizers.Adam(learning_rate=0.00003)
model.compile(loss='mean_squared_error', optimizer=opt)

model.fit(tfx, epochs=60, callbacks = my_callbacks)

model.save('agil_sttr_32.h5')

