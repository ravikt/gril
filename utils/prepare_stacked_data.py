'''
The script prepares data for gaze regularized imitation learning
in the following format
[rgb, depth, action, gaze_coord]
'''


import argparse
import csv
import cv2
import numpy as np
import os
import re
from itertools import zip_longest

def reshape_depth(depth):
    """Resize to the same size as the one in Ritwik's work."""
    width = 224
    height = 224
    frame = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    frame = np.expand_dims(frame, axis=2)
    #cv2.imwrite('rgb_cv.png', frame)
    #plt.imsave('rgb_plt.png', frame)
    return frame / 255.0

def reshape_image(image):
    """Resize to the size as required in ResNet."""
    width = 224
    height = 224
    frame = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    return frame / 255.0


def prepare_data(data_path):
    for subdir in os.listdir(data_path):
       print(subdir)
       fname = ["rgb", "log.csv"]
       dirname = os.path.join(data_path, subdir)
       print(dirname)

       imgs = []
       depth= []
       gaze_pos = []
       act_lbls = []
        # img_idx=[]
        # print(os.path.join(dirname, fname)
       if fname[1].endswith('.csv'):
          log = fname[1]
       csv_path = os.path.join(dirname, log)
       print(csv_path)
       with open(csv_path, 'r') as csvfile:
            gaze = csv.reader(csvfile)
            next(gaze)  # skip the head rowi

            for i, row in enumerate(zip_longest(gaze, gaze)):
                if row[1]:
                   # read consecutive images
                   img_path1   = os.path.join(dirname, "rgb", row[0][3].split("/")[-1])
                   img_path2   = os.path.join(dirname, "rgb", row[1][3].split("/")[-1])
     
                   # read consecutive depth images
                   depth_path1 = os.path.join(dirname, "depth", row[0][5].split("/")[-1])                
                   depth_path2 = os.path.join(dirname, "depth", row[1][5].split("/")[-1])                

                   # print(img_path)
                   print("rgb", row[0][3].split("/")[-1], row[1][3].split("/")[-1])
                    
                    # read color images
                   im1 = np.float32(cv2.imread(img_path1))
                   im1 = reshape_image(im1)  
                   im2 = np.float32(cv2.imread(img_path2))
                   im2 = reshape_image(im2)  

                   imgs.append(np.dstack((im1, im2)))
                     
                    # read depth images
                   dt1 = np.float32(cv2.imread(depth_path1))
                   dt1 = reshape_depth(dt1)
                   dt2 = np.float32(cv2.imread(depth_path2))
                   dt2 = reshape_depth(dt2)
                    
                   depth.append(np.dstack((dt1, dt2)))

                    # gaze coordinate
                    # coords1 = np.array((row[0][-3], row[0][-2]))
                    # coords2 = np.array((row[1][-3], row[1][-2]))
                   coords1 = np.array((float(row[0][-3]), float(row[0][-2])))
                   coords2 = np.array((float(row[1][-3]), float(row[1][-2])))
                    # coords = np.hstack((coords1, coords2))
                   coords = np.mean([coords1, coords2], axis=0)
                   gaze_pos.append(coords)
                    
                    # action labels
                    # act_roll, act_pitch, act_throttle, act_yaw
                   act_commands1 = np.array((float(row[0][-7]), float(row[0][-6]), float(row[0][-5]), float(row[0][-4])))
                   act_commands2 = np.array((float(row[1][-7]), float(row[1][-6]), float(row[1][-5]), float(row[1][-4])))
                    # act_commands = np.hstack((act_commands1, act_commands2))
                   act_commands = np.mean([act_commands1, act_commands2], axis=0)
                   act_lbls.append(act_commands)

            gaze_pos = np.array(gaze_pos)
            #gaze_pos = gaze_pos.astype(float)

            act_lbls = np.array(act_lbls)
            #act_lbls = act_lbls.astype(float)
            # print(gaze_pos.shape)
            imgs = np.array(imgs)
            depth= np.array(depth)
            #print(imgs.shape)


            
            print(depth.shape, depth.dtype)
            print(imgs.shape, imgs.dtype)
            print(gaze_pos.shape, gaze_pos.dtype)
            print(act_lbls.shape, act_lbls.dtype)
             
            gaze_pos = np.reshape(gaze_pos, (gaze_pos.shape[0], gaze_pos.shape[1], 1))
            act_lbls = np.reshape(act_lbls, (act_lbls.shape[0], act_lbls.shape[1], 1))
            
       npz_name = data_path.split("/")[-2]
       print(npz_name)
       np.savez_compressed(f"{npz_name}.npz", images=imgs, depth=depth, action=act_lbls, gaze_coords=gaze_pos)


if __name__ == "__main__":
    #data_path = '/scratch/user/ravikt/airsim/data/moving_truck_mountains3/'
    parser = argparse.ArgumentParser(description="script for creating numpy compressed data")
    parser.add_argument("-p", "--path", type=str, help="path to airsim data")

    args = parser.parse_args()
    data_path = args.path
    prepare_data(data_path)
                             
