# Script of during rollout with AirSim binaries

import argparse
import cv2
import tensorflow as tf
import numpy as np
#from tf.keras.models import load_model
from agil_gaze import my_softmax, my_kld
import matplotlib.pyplot as plt
import os
import re
import datetime

#from utils import read_gaze


def reshape_image(image):
    """Resize to the same size as the one in Ritwik's work."""
    width = 224
    height = 224
    
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    frame = np.expand_dims(frame, axis=2)
    frame = np.expand_dims(frame, axis=0)
    #cv2.imwrite('rgb_cv.png', frame)
    #plt.imsave('rgb_plt.png', frame)
    return frame / 255.0



customObjects = {
    'my_softmax': my_softmax,
    'my_kld': my_kld,
}


def agilNN(gaze_model, agil_model, airsim_img):#, i):
    # Reshape image from AirSim camera
    img = reshape_image(airsim_img)
    # plt.axis('off')
    # plt.imshow(np.squeeze(ghmap, axis=0))
    # plt.savefig(f'gh_{i}.png', transparent=True, bbox_inches='tight')
    print("Printing image shape")
    #print(img.shape)
    ghmap = gaze_model.predict(img)
    #ghmap = gaze_model(img)

    # plt.axis('off')
    # plt.imshow(np.squeeze(ghmap, axis=0))
    # #print(ghmap.shape)
    # plt.savefig(f'gh_{i}.png', transparent=True, bbox_inches='tight')
    
    input_agil = [img, ghmap]
    commands = agil_model.predict(input_agil)
    #commands = agil_model(input_agil)
    print("Commands", commands)
    return commands

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
