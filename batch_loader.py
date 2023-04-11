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
        train_depth = data['depth']
        #print(train_depth.shape)
        train_act = data['action']
        print(train_act.shape)
        train_gaze = data['gaze_coords']
        #train_gaze = data['heatmap']
        #print(train_gaze.shape)
        return train_imgs, train_depth, train_act, train_gaze

def generate_gril(path, file_list):
    # Generate batches of samples
    #while 1:
        # indexes = folder_list
        imax = int(len(file_list)/1)  # 1,2,...number of npz files
        for i in range(imax):
            file_npz = [k for k in file_list[i*1:(i+1)*1]]
            #file_npz = file_list[i*1:(i+1)*1]

            print(file_npz)
            
            file_npz = b" ".join(file_npz)
            #file_npz = " ".join(file_npz)
   
            imgs, depth, acts, gaze = read_npz(os.path.join(path, file_npz))
            #X, y = read_npz(file_npz)
            #print(img.dtype, gaze.dtype)
            for idx in range(0, len(depth)):

                #yield {"input_1":img, "input_2":gaze}, act
                yield {"image": imgs[idx], "depth": depth[idx]}, {"action":acts[idx], "gaze":gaze[idx]}
                

def generate_il_cgl(path, file_list):
    # Generate batches of samples
    #while 1:
        # indexes = folder_list
        imax = int(len(file_list)/1)  # 1,2,...number of npz files
        for i in range(imax):
            file_npz = [k for k in file_list[i*1:(i+1)*1]]
            #file_npz = file_list[i*1:(i+1)*1]

            print(file_npz)

            file_npz = b" ".join(file_npz)
            #file_npz = " ".join(file_npz)

            imgs, depth, acts, gaze = read_npz(os.path.join(path, file_npz))
            #X, y = read_npz(file_npz)
            #print(img.dtype, gaze.dtype)
            for idx in range(0, len(depth)):
                
                #image = cv2.cvtColor(imgs[idx], cv2.COLOR_BGR2GRAY)
                #print(gaze[idx].shape)
                gaze_reshaped = cv2.resize(gaze[idx], (28, 28), interpolation=cv2.INTER_AREA) 
                #yield {"input_1":img, "input_2":gaze}, act
                yield {"image": imgs[idx]},  {"gaze":gaze_reshaped, "action":acts[idx]}




