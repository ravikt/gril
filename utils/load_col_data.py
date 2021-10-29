import csv
import cv2
import numpy as np
import os
import re
from read_gaze import preprocess_gaze_heatmap, reshape_heatmap, reshape_image


def get_im_index(filename):
    return re.search(r'\d+', filename).group(0)


def prepare_data(data_path):
    for item in os.listdir(data_path):
        # Get the top level child sub-directory
        print(item)
        imgs = []
        gaze_pos = []
        # img_idx=[]
        for root, dirs, images in os.walk(os.path.join(data_path, item)):
            for imname in images:
                if imname.endswith('.csv'):
                    log = imname

        csv_path = os.path.join(data_path, item, log)
        with open(csv_path, 'r') as csvfile:
            gaze = csv.reader(csvfile)
            next(gaze)  # skip the head row
            for i, row in enumerate(gaze):
                img_path = os.path.join(data_path, item, "rgb", f"rgb_{i}.png")
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
            print(imgs.shape)

            hmap = preprocess_gaze_heatmap(np.array(gaze_pos), 3)
            hmap = np.squeeze(hmap, axis=1)
            hmap = reshape_heatmap(hmap)
            print(hmap.shape)
            print(len(hmap))

        np.savez_compressed(f"{item}.npz", images=imgs, heatmap=hmap)


if __name__ == "__main__":
    data_path = '/home/ravi/data/'
    prepare_data(data_path)
