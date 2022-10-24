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


def get_im_index(filename):
    return re.search(r'\d+', filename).group(0)

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
            next(gaze)  # skip the head row
            for i, row in enumerate(gaze):
                img_path   = os.path.join(dirname, "rgb", f"rgb_{i}.png")
                depth_path = os.path.join(dirname, "depth", f"depth_{i}.png")                
                # print(img_path)
                if os.path.exists(img_path):
                    print(f"rgb_{i}.png", row[-3], row[-2])
                    
                    # read color images
                    # flip the image horizontally 
                    im = np.float32(cv2.flip(cv2.imread(img_path), 1))
                    im = reshape_image(im)  
                    imgs.append(im)
                     
                    # read depth images
                    # flip the depth image horizontally
                    dt = np.float32(cv2.flip(cv2.imread(depth_path), 1))
                    dt = reshape_depth(dt)
                    depth.append(dt)

                    # gaze coordinate
                    # coords = np.hstack((row[-3], row[-2]))
                    # gaze cordinates flipped
                    coords = np.hstack((1.0-float(row[-3]), float(row[-2])))
                    gaze_pos.append(coords)
                    
                    # action labels
                    # control commands fized
                    act_commands = np.hstack((-float(row[-7]), float(row[-6]), float(row[-5]),-float(row[-4])))
                    act_lbls.append(act_commands)

            gaze_pos = np.array(gaze_pos)
            gaze_pos = gaze_pos.astype(float)

            act_lbls = np.array(act_lbls)
            act_lbls = act_lbls.astype(float)
            # print(gaze_pos.shape)
            imgs = np.array(imgs)
            depth= np.array(depth)
            #print(imgs.shape)

      
            print(depth.shape)
            print(imgs.shape)
            print(gaze_pos.shape)
            print(act_lbls.shape)
            #depth    = np.reshape(depth, (depth.shape[0], depth.shape[1], depth.shape[2], 1))
            #imgs     = np.reshape(imgs, (imgs.shape[0], imgs.shape[1], imgs.shape[2], 1)) 
            gaze_pos = np.reshape(gaze_pos, (gaze_pos.shape[0], gaze_pos.shape[1], 1))
            act_lbls = np.reshape(act_lbls, (act_lbls.shape[0], act_lbls.shape[1], 1))
       npz_name = data_path.split("/")[-2]
       print(npz_name)
       np.savez_compressed(f"flipped_{npz_name}.npz", images=imgs, depth=depth, action=act_lbls, gaze_coords=gaze_pos)


if __name__ == "__main__":
    #data_path = '/scratch/user/ravikt/airsim/data/moving_truck_mountains3/'
    parser = argparse.ArgumentParser(description="script for creating numpy compressed data")
    parser.add_argument("-p", "--path", type=str, help="path to airsim data")

    args = parser.parse_args()
    data_path = args.path
    prepare_data(data_path)
                             
