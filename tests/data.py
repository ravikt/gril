'''
My earlier assumption was that tf.Dataset eliminates the need for 
creating generator function. This script was an attempt to 
build a data pipeline using native tf dataset API.
It was supposed to load mutiple npz files and pass it in batches.
'''

import numpy as np 
#import tensorflow as tf
import cv2, os

def load_data(path):
    # path = "/scratch/user/ravikt/small.npz"
    with np.load(path) as data:
        l = len(data["images"])
        train_images = np.reshape(data['images'], (l, 224, 224, 1))
        train_gaze = np.reshape(data['heatmap'], (l, 224, 224, 1))
        # train_lbls
    return train_images, train_gaze

   
dirs = "/scratch/user/ravikt/airsim/test_data/"

def npz_path(dir_path):
    '''Function to yield npz filename'''
    file_list = iter(os.listdir(dirs))
    # may be call load_data somewhere here
    yield file_list

name = npz_path(dirs)

print(next(name))


