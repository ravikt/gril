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
import pandas as pd

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

        imgs_flip = []
        depth_flip= []
        gaze_pos_flip = []
        act_lbls_flip = []
        # img_idx=[]
        # print(os.path.join(dirname, fname)
        if fname[1].endswith('.csv'):
           log = fname[1]
        csv_path = os.path.join(dirname, log)
        print(csv_path)
        train_df = pd.read_csv(csv_path)
        train_df["rgb_addr"]  = train_df["rgb_addr"].apply(lambda x: x.split("/")[-1])
        train_df["depth_addr"]= train_df["depth_addr"].apply(lambda x: x.split("/")[-1])
        
        non_zero_yaw = train_df[train_df["act_yaw"] != 0.0]
        zero_yaw = train_df.query("act_yaw == 0.0").sample(frac=0.10)
        final_df = pd.concat([zero_yaw, non_zero_yaw])

        for i, data in final_df.iterrows():
            img_path   = os.path.join(dirname, "rgb", data["rgb_addr"])
            depth_path = os.path.join(dirname, "depth", data["depth_addr"])                
            # print(img_path)
            # if (os.path.exists(img_path) and float(row[-4])==0.0):
            print(data["rgb_addr"], data["gaze_x"], data["gaze_y"])

            # flip the image horizontally 
            im_flip = np.float32(cv2.flip(cv2.imread(img_path), 1))
            im_flip = reshape_image(im_flip)  
            imgs_flip.append(im_flip)

            # flip the depth image horizontally
            dt_flip = np.float32(cv2.flip(cv2.imread(depth_path), 1))
            dt_flip = reshape_depth(dt_flip)
            depth_flip.append(dt_flip)

            # gaze cordinates flipped
            coords_flip = np.hstack((1.0-data["gaze_x"], data["gaze_y"]))
            gaze_pos_flip.append(coords_flip)
            
            # control commands fized
            act_commands_flip = np.hstack((-1.0*data["act_roll"], data["act_pitch"], data["act_throttle"], -1.0*data["act_yaw"]))
            act_lbls_flip.append(act_commands_flip)

        gaze_pos_flip = np.array(gaze_pos_flip)
        gaze_pos_flip = gaze_pos_flip.astype(float)

        act_lbls_flip = np.array(act_lbls_flip)
        act_lbls_flip = act_lbls_flip.astype(float)
        # print(gaze_pos.shape)
        imgs_flip = np.array(imgs_flip)
        depth_flip = np.array(depth_flip)
        #print(imgs.shape)

    
        print(depth_flip.shape)
        print(imgs_flip.shape)
        print(gaze_pos_flip.shape)
        print(act_lbls_flip.shape)

        gaze_pos_flip = np.reshape(gaze_pos_flip, (gaze_pos_flip.shape[0], gaze_pos_flip.shape[1], 1))
        act_lbls_flip = np.reshape(act_lbls_flip, (act_lbls_flip.shape[0], act_lbls_flip.shape[1], 1))

        npz_name = data_path.split("/")[-2]
        print(npz_name)
        np.savez_compressed(f"flipped_{npz_name}.npz", images=imgs_flip, depth=depth_flip, action=act_lbls_flip, gaze_coords=gaze_pos_flip)



if __name__ == "__main__":
    #data_path = '/scratch/user/ravikt/airsim/data/moving_truck_mountains3/'
    parser = argparse.ArgumentParser(description="script for creating numpy compressed data")
    parser.add_argument("-p", "--path", type=str, help="path to airsim data")

    args = parser.parse_args()
    data_path = args.path
    prepare_data(data_path)
                             

