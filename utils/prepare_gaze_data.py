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
       allow = ["rgb", "log.csv"]
       dirname = os.path.join(data_path, subdir)
       print(dirname)
       # This loop is causing the pre processing to run twice
       # Do something about it
       for fname in os.listdir(dirname):
          if fname in allow:

            # Get the top level child sub-directory
            print(fname)
            imgs = []
            gaze_pos = []
            # img_idx=[]
            # print(os.path.join(dirname, fname)
            if fname.endswith('.csv'):
               log = fname
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
    
                gaze_pos = np.array(gaze_pos)
                gaze_pos = gaze_pos.astype(float)
                # print(gaze_pos.shape)
                imgs = np.array(imgs)
                #print(imgs.shape)
    
                hmap = preprocess_gaze_heatmap(np.array(gaze_pos), 3)
                hmap = np.squeeze(hmap, axis=1)
                hmap = reshape_heatmap(hmap)
                #print(hmap.shape)
                #print(len(hmap))
    
            np.savez_compressed(f"{subdir}.npz", images=imgs, heatmap=hmap)


if __name__ == "__main__":
    data_path = '/scratch/user/ravikt/col_data/moving_truck_mountains14/'
    #import argparse
    #parser = argparse.ArgumentParser(description="Model prediction script")
    #parser.add_argument("-p", "--path", type=str, help="path to raw dataset")
   
    #args = parser.parse_args()
    #data_path = args.path
    prepare_data(data_path)
