

import argparse
import cv2
import sys
import numpy as np
from load_data import Dataset
import matplotlib.pyplot as plt
import os
import re
import datetime

def overlay(img, ghmap):
    output = cv2.addWeighted(ghmap, 0.4, img, 0.6, 0.0)
    return output

def reshape_image(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width = 224
    height = 224
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame
    #return frame / 255.0 

def video(img_dir, ghmap_dir):
    img_files = os.listdir(img_dir)
    img_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    ghmap_files = os.listdir(ghmap_dir) 
    ghmap_files.sort(key=lambda g: int(re.sub('\D', '', g)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('video.avi', fourcc, 1, (224, 224))

    for (img_path,ghmap_path) in zip(img_files, ghmap_files):
       image = cv2.imread(os.path.join(img_dir, img_path))
       image = np.uint8(reshape_image(image))
       print(image.shape)

       ghmap = cv2.imread(os.path.join(ghmap_dir, ghmap_path))
       output = overlay(image, ghmap)
       video.write(output)

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for writing videos")
    parser.add_argument("-i", "--inpath", type=str, help="path to the image folder")   
    parser.add_argument("-g", "--ghpath", type=str, help="path to the gaze golder") 
    args = parser.parse_args()
    # Path for model and the data
    input_path = args.inpath
    gaze_path  = args.ghpath
    video(input_path, gaze_path)
