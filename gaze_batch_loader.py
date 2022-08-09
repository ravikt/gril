import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import os
import glob
import random

def read_npz(data_path):
    with np.load(data_path) as data:
        l = len(data["images"])
        train_imgs = data['images']
        print(train_imgs.shape)
        train_gaze = data['gaze_coords']
        print(train_gaze.shape)
        return train_imgs, train_gaze

def generate_gaze(path, file_list):
    # Generate batches of samples
    # The following while loop will make the the raining stuck 
    # at Epoch 1 (Useful for testing generator)
    # while 1:
        # indexes = folder_list
        imax = int(len(file_list)/1)  # 1,2,...number of npz files
        for i in range(imax):
            file_npz = [k for k in file_list[i*1:(i+1)*1]]

            print(file_npz)
            # The below hack converts bytes to string
            # After spending three hours found this reference
            # https://stackoverflow.com/questions/32071536/typeerror-sequence-item-0-expected-str-instance-bytes-found
            file_npz = b" ".join(file_npz)
            # Following does not work with tensorflow
            #file_npz = " ".join(file_npz)
   
            imgs, gaze = read_npz(os.path.join(path, file_npz))
        
            for idx in range(0, len(gaze)):
                # If batch_size is given in training code
                print(idx, imgs[idx].shape, gaze[idx].shape)
                yield imgs[idx], gaze[idx]
                # If bacth size is not specified in the training code
                #yield np.reshape(imgs[idx], (1, 224, 224, 1)), np.reshape(ghmaps[idx], (1, 224, 224, 1))



if __name__ == "__main__":
    # test the generator function
    datapath = "/scratch/user/ravikt/airsim/debug_data/"
    file_list = os.listdir(datapath)
    print(len(file_list))
    random.shuffle(file_list)
    #steps_per_epoch = np.int(np.ceil(len(file_list)/batch_size))
    x = generate_gaze(datapath, file_list)
    tfx = tf.data.Dataset.from_generator(generate_gaze,args=[datapath, file_list], output_types = (tf.float32, tf.float32))    
    img_counter = 0
    for a,b in tfx:
        print(img_counter)
        print(a.dtype, b.shape)
        #cv2.imwrite('rgb{}.png'.format(img_counter), a)
        plt.imsave('rgb{}.png'.format(img_counter), np.squeeze(a))

        img_counter = img_counter+1
    #tfx = tf.data.Dataset.from_generator(generate_gaze,args=[datapath, file_list], output_types = (tf.float32, tf.float32))
 
