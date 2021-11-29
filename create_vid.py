# Example python predict.py -p ../results_test/small_large_res.h5 -i ../sample/img/ -o ../sample/

import argparse
import cv2
import sys
import numpy as np
from load_data import Dataset
import matplotlib.pyplot as plt
import os
import re
import datetime


def reshape_image(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width = 224
    height = 224
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame / 255.0


customObjects = {
    'my_softmax': my_softmax,
    'my_kld': my_kld,
    'NSS': NSS
}


#d = Dataset(sys.argv[1], sys.argv[2])
#sample = d.generate_data_for_gaze_prediction()

# d.load_predicted_gaze_heatmap(sys.argv[3])


def save_preds(idx, img_pred, ouput_path):
    now = datetime.datetime.now()
    save_dir = os.path.join(output_path, f'{now.month}{now.day}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.imshow(np.squeeze(img_pred))
    plt.imsave(os.path.join(save_dir, f'gh_{idx}.png'), np.squeeze(img_pred))


def get_index(filename):
    return re.search(r'\d+', filename).group(0)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model prediction script")
    parser.add_argument("-i", "--inpath", type=str, help="path to test folder")
    parser.add_argument("-g", "--ghpath", type=str, help="path to gaze heatmap")
    parser.add_argument("-o", "--outpath",type=str, help="path to video path")
    
    args = parser.parse_args()
    # Path for model and the data
    model_path = args.path
    input_path = args.inpath
    output_path = args.outpath
    
    gaze_predict(input_path, model_path, output_path)
