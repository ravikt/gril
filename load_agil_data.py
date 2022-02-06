'''
Data loader for Airsim dataset
This file reads dataset by
Bera et al Gaze augmented imitation learning
'''

import sys, os, re, threading, time, copy
import numpy as np
import csv
import cv2
from utils.read_gaze import reshape_heatmap, reshape_image


def reshape(img):
    '''
    Reshape the rgb and gaze images to (n, n, 1)
    '''
    width = 224 #84
    height = 224 #84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    frame = np.reshape(np.array(frame), (224, 224, 1))
    return frame / 255.0

def action_label(csv_path, dirname):
    act_lbls = []
    with open(csv_path, 'r') as csvfile:
        act = csv.reader(csvfile)
        next(act)  # skip the head row
        for i, row in enumerate(act):
            img_path = os.path.join(dirname, f"rgb_{i}.png")
            # print(img_path)
            if os.path.exists(img_path):
                print(f"rgb_{i}.png", row[-7], row[-6], row[-5], row[-4])

                action_coords = np.hstack((row[-7], row[-6], row[-5], row[-4]))
                act_lbls.append(action_coords)

        act_lbls = np.array(act_lbls)
        act_lbls = act_lbls.astype(float)
    np.savez_compressed(f"act_tm2.npz", action=act_lbls)
    #return act_lbls


def some_loader(imgs_dir, csv_dir, gaze_dir):
    '''
    The function takes directory path of images, gaze and log files
    It retruns the resized images, gaze heatmap and action label as
    numpy array.
    '''
    # list all the image files and sort
    # in order of their index
    imgs_list =  os.listdir(imgs_dir)
    imgs_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    gaze_list = os.listdir(gaze_dir)
    gaze_list.sort(key=lambda g: int(re.sub('\D', '', g)))

    lbls_stack = action_label(csv_dir, imgs_dir)
    imgs_stack = []
    gaze_stack = []

    for img, gaze in zip(imgs_list, gaze_list):
        img_path = os.path.join(imgs_dir, img)
        gaze_path= os.path.join(gaze_dir, gaze)

        image = np.float32(cv2.imread(img_path))
        image = reshape_image(image)
        imgs_stack.append(image)

        ghmap = cv2.imread(gaze_path)
        ghmap = reshape_image(ghmap)
        gaze_stack.append(ghmap)

    imgs_stack = np.array(imgs_stack)
    gaze_stack = np.array(gaze_stack)
    
    return imgs_stack, lbls_stack, gaze_stack

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="AGIL Network Architecture")

    parser.add_argument("-i", "--images", type=str, help="path to images" )
    parser.add_argument("-l", "--labels", type=str, help="path to action labels")
    #parser.add_argument("-g", "--ghmap",  type=str, help="path to predicted gaze heatmap")

    args = parser.parse_args()

    train_imgs_path = args.images
    train_lbls_path = args.labels
    #train_gaze_path = args.ghmap
    action_label(train_lbls_path, train_imgs_path)
    #train_imgs, train_lbls, train_gaze = some_loader(train_imgs_path, train_lbls_path, train_gaze_path)

    #print(train_imgs.shape)
    #print(train_gaze.shape)
    #print(train_lbls.shape)
