

import argparse
import cv2
import sys
import numpy as np
from load_data import Dataset
import matplotlib.pyplot as plt
import os
import re
import datetime

def overlay():
    pass

def reshape_image(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width = 224
    height = 224
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame
    #return frame / 255.0 

def video(img_dir):
    files = os.listdir(img_dir)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('video.avi', fourcc, 1, (224, 224))

    for img_path in files:
       image = cv2.imread(os.path.join(img_dir, img_path))
       image = np.uint8(reshape_image(image))
       print(image.shape)
       video.write(image)

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for writing videos")
    parser.add_argument("-i", "--inpath", type=str, help="path to the folder")   
    args = parser.parse_args()
    # Path for model and the data
    input_path = args.inpath
    
    video(input_path)
