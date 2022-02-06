'''
My earlier assumption was that tf.Dataset eliminates the need for 
creating generator function. This script was an attempt to 
build a data pipeline using native tf dataset API.
It was supposed to load mutiple npz files and pass it in batches.
'''

import numpy as np 
import tensorflow as tf
import cv2, os

def load_data(path):
    # path = "/scratch/user/ravikt/small.npz"
    with np.load(path) as data:
        l = len(data["images"])
        train_images = np.reshape(data['images'], (l, 224, 224, 1))
        train_gaze = np.reshape(data['heatmap'], (l, 224, 224, 1))
        # train_lbls
    return train_images, train_gaze

@tf.function
def tf_load_path(npz_path):
    imgs, gaze = tf.py_function(load_data, [npz_path], tf.float32)
    return imgs, gaze
    

dirs = "/scratch/user/ravikt/airsim/test_data/"
file_list = os.listdir(dirs)

#list_ds = tf.data.Dataset.list_files(file_list)

dataset = tf.data.Dataset.from_tensor_slices(file_list)
#dataset = dataset.shuffle(len(file_list))
dataset = dataset.map(tf_load_path)
#dataset = dataset.batch(1)
#dataset = dataset.prefetch(1)

#for i in dataset:
#   print(i.numpy())
