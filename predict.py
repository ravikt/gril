''' 
This scripts generates prediction from gaze model(.h5 file). It takes 224x224 normalized image
and produces gaze heatmap. Its purpose is to verify the trained gaze models and 
visualize heatmap results.


Example usage:
              python predict.py -p <path/to/model.h5/file> -i ..</sample/img/> -o ..</output/directory/for/results>
'''

import argparse
import cv2
import tensorflow as tf
import sys
import numpy as np
from losses import my_softmax, my_kld
import matplotlib.pyplot as plt
import os
import re
import datetime


def reshape_image(image):
    """Warp frames to 224x224 as done in Ritwik's thesis."""
    width = 224 #84
    height = 224 #84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    frame = np.expand_dims(frame, axis=2)
    frame = np.expand_dims(frame, axis=0)
    return frame / 255.0



customObjects = {
    'my_softmax': my_softmax,
    'my_kld': my_kld
}


def save_preds(idx, img_pred, ouput_path):
    now = datetime.datetime.now()
    save_dir = os.path.join(output_path, f'{now.month}{now.day}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.imshow(np.squeeze(img_pred))
    plt.imsave(os.path.join(save_dir, f'gh_{idx}.png'), np.squeeze(img_pred))


def get_index(filename):
    return re.search(r'\d+', filename).group(0)


def gaze_predict(input_path, model_path, output_path):
    dirs = os.listdir(input_path)
    # Load the trained model
    print("Predicting results...")
    agil = tf.keras.models.load_model(model_path, custom_objects=customObjects)
    # agil.summary()
    for img_path in sorted(dirs):
        print(img_path)
        img = cv2.imread(os.path.join(input_path, img_path))
        #print(img.shape)
        #print(os.path.join(test_data_path, img_path))
        img = reshape_image(img)
        print(img.shape)
        #img = np.reshape(img, (1, 224, 224, 1))

        output = agil.predict(img, batch_size=1)
        print(output.shape)
        out_img = np.squeeze(output, axis=0)
        save_preds(get_index(img_path), out_img, output_path)
    print("Prediction complete!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model prediction script")
    parser.add_argument("-p", "--path", type=str, help="path to model file")
    parser.add_argument("-i", "--inpath", type=str, help="path to test folder")
    parser.add_argument("-o", "--outpath", type=str, help="path for results")
    
    args = parser.parse_args()
    # Path for model and the data
    model_path = args.path
    input_path = args.inpath
    output_path = args.outpath
    
    gaze_predict(input_path, model_path, output_path)
