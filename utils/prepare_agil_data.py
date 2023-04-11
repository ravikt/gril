import argparse
import csv
import cv2
import numpy as np
import os
import re
from read_gaze import preprocess_gaze_heatmap, reshape_heatmap, reshape_image


def get_im_index(filename):
    return re.search(r'\d+', filename).group(0)

def prepare_data(data_path):
    for subdir in os.listdir(data_path):
       print(subdir)
       fname = ["rgb", "log.csv"]
       dirname = os.path.join(data_path, subdir)
       print(dirname)

       imgs = []
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
                img_path = os.path.join(dirname, "rgb", f"rgb_{i}.png")
                # print(img_path)
                if os.path.exists(img_path):
                    print(f"rgb_{i}.png", row[-3], row[-2])
                    im = np.float32(cv2.imread(img_path))
                    im = reshape_image(im)  # reshape to 84x84
                    imgs.append(im)

                    coords = np.hstack((row[-3], row[-2]))
                    gaze_pos.append(coords)
                    act_commands = np.hstack((row[-7], row[-6], row[-5], row[-4]))
                    act_lbls.append(act_commands)

            gaze_pos = np.array(gaze_pos)
            gaze_pos = gaze_pos.astype(float)

            act_lbls = np.array(act_lbls)
            act_lbls = act_lbls.astype(float)
            # print(gaze_pos.shape)
            imgs = np.array(imgs)
            #print(imgs.shape)

            hmap = preprocess_gaze_heatmap(np.array(gaze_pos), 10)
            hmap = np.squeeze(hmap, axis=1)
            hmap = reshape_heatmap(hmap)
            print(hmap.shape)
            print(imgs.shape)
            print(act_lbls.shape)
            hmap = np.reshape(hmap, (hmap.shape[0], hmap.shape[1], hmap.shape[2], 1))  
            imgs = np.reshape(imgs, (imgs.shape[0], imgs.shape[1], imgs.shape[2], 1)) 
            act_lbls = np.reshape(act_lbls, (act_lbls.shape[0], act_lbls.shape[1], 1))
       npz_name = data_path.split("/")[-2]
       print(npz_name)
       np.savez_compressed(f"{npz_name}.npz", images=imgs, heatmap=hmap, vel_comm=act_lbls)


if __name__ == "__main__":
    #data_path = '/scratch/user/ravikt/airsim/data/moving_truck_mountains3/'
    parser = argparse.ArgumentParser(description="script for creating numpy compressed data")
    parser.add_argument("-p", "--path", type=str, help="path to airsim data")

    args = parser.parse_args()
    data_path = args.path
    prepare_data(data_path)
                             
