''' 
This script performs offline prediction of control commands
from images of AirSim trejectories. 


Example usage:
              python off_eval.py -p <path/to/model.h5/file> -i ..</sample/img/> -l ..</path/to/logfile/>
'''

import argparse
import cv2
import tensorflow as tf
import sys
import numpy as np
from losses import my_softmax, my_kld
import matplotlib.pyplot as plt
import os
import re
import datetime
import csv
from itertools import zip_longest



def reshape(image):
    """Warp frames to 224x224 as Resnet's input size."""
    width = 224 #84
    height = 224 #84
    frame = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return frame / 255.0

def bc(input_path, model_path, logfile):
    roll = []
    pitch = []
    throttle = []
    yaw = []
    with open(logfile, 'r') as csvfile:
        gaze = csv.reader(csvfile)
        next(gaze)  # skip the head row
        bc = tf.keras.models.load_model(model_path)
        for i, row in enumerate(zip_longest(gaze, gaze)):
            img = []
            if row[1]:
              im1 = row[0][3].split("/")[-1]
              im2 = row[1][3].split("/")[-1]
              
              img1 = cv2.imread(os.path.join(input_path, im1))
              img2 = cv2.imread(os.path.join(input_path, im2))
     
              img1 = np.float32(reshape(img1))
              img2 = np.float32(reshape(img2))
            
              img = np.dstack((img1, img2))
              img = np.expand_dims(img, axis=0) 
              print(i,img.dtype, row[0][3].split("/")[-1],  row[1][3].split("/")[-1], img.shape)  
              commands = bc.predict(img)
              print(commands)
              roll.append(commands[0][0])
              pitch.append(commands[0][1])
              throttle.append(commands[0][2])
              yaw.append(commands[0][3])
              
        print("prediction complete")
        print(commands[0][1], commands[0][2]) 
        roll    = np.array(roll)
        pitch   = np.array(pitch)
        throttle= np.array(throttle)
        yaw     = np.array(yaw) 
    print(roll.dtype, roll.shape)
    plot_commands(roll, pitch, throttle, yaw)       

def plot_commands(roll, pitch, throttle, yaw):
    
    plt.figure()
    plt.plot(yaw)
    plt.title('Yaw')
    plt.savefig("yaw.png")


    plt.figure()
    plt.plot(throttle, 'g')
    plt.title("Throttle")
    plt.savefig("throttle.png")


    plt.figure()
    plt.plot(pitch, 'r')
    plt.title("Pitch")
    plt.savefig("pitch.png")


    plt.figure()
    plt.plot(roll, 'c')
    plt.title("Roll")
    plt.savefig("roll.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model prediction script")
    parser.add_argument("-p", "--path", type=str, help="path to model file")
    parser.add_argument("-i", "--inpath", type=str, help="path to test folder")
    parser.add_argument("-l", "--logfile", type=str, help="path to log file")
    
    args = parser.parse_args()
    # Path for model and the data
    model_path = args.path
    input_path = args.inpath
    logfile = args.logfile
    
    bc(input_path, model_path, logfile)
