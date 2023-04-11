# Script of during rollout with AirSim binaries

import argparse
from imaplib import Commands
import cv2
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import os
import re
import datetime

def reshape_depth(depth):
    """Resize to the same size as the one in Ritwik's work."""
    width = 224
    height = 224
    frame = depth
    # frame = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    frame = np.expand_dims(frame, axis=2)
    frame = np.expand_dims(frame, axis=0)
    return frame / 255.0


def reshape_image(image):
    """Resize to the size as required in ResNet."""
    width = 224
    height = 224
    frame = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    frame = np.expand_dims(frame, axis=0)
    return frame / 255.0


def arilNN(airsim_img, depth, aril):
    # Reshape image from AirSim camera
    print("Type and max-min values")
    print(airsim_img.dtype, depth.dtype)
    print(np.max(airsim_img), np.min(airsim_img))
    print(np.max(depth), np.min(depth))

    img = reshape_image(np.float32(airsim_img))
    
    depth = reshape_depth(np.float32(depth))
    # print("Printing image shape")
    # print(img.shape)
    # print(depth.shape)
    # print("max and min value after reshaping")
    # print(img.dtype, depth.dtype)
    # print(np.max(img), np.min(img))
    # print(np.max(depth), np.min(depth))
    
    input_data = [img, depth]
    commands, gaze = aril.predict(input_data)
    #commands = agil_model(input_agil)
    print("Commands", commands)
    return commands, gaze

if __name__ == "__main__":
    pass
    # parser = argparse.ArgumentParser(description="Model prediction script")
    # parser.add_argument("-g", "--gaze", type=str, help="path to gaze model")
    # parser.add_argument("-a", "--agil", type=str, help="path to agil model")
    
    # args = parser.parse_args()
    # Path for model and the data
    # gaze_path = args.gaze
    # agil_path = args.agil
    
    #gaze_predict(input_path, model_path, output_path)
