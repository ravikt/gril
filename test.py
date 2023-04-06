# Modified by Sunbeam, Ravi (2023)
import sys, os, re, threading, time, copy
import numpy as np
import tarfile
import cv2
import pandas as pd
import tensorflow as tf


def preprocess(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width = 84
    height = 84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame / 255.0

# label_fname = "/home/grads/m/mdsunbeam/atari_head/IL-CGL/data/alien/314_RZ_9847886_Jun-06-14-05-27.txt"
# # Read action labels from txt file
# # train_df = pd.read_csv(label_fname, delimiter="\t")
# frame_ids, lbls, x, y= [], [], [], []

# # print(train_df)
# # # for i, data in train_df.iterrows():
        
# # #         # if (os.path.exists(img_path) and float(row[-4])==0.0):
# # #         print(data["episode_id"])

# with open(label_fname,'r') as f:
#     for line in f:
#         line = line.strip()
#         if line.startswith("frame_id") or line == "": 
#             continue # skip head or empty lines
#         dataline = line.split(',') 
#         if dataline[6] == "null":
#             frame_id, lbl, gaze_x, gaze_y = dataline[0], dataline[5], -1, -1
#         else:
#             frame_id, lbl, gaze_x, gaze_y = dataline[0], dataline[5], dataline[6], dataline[7]
#         if lbl == "null": # end of file
#             break
#         frame_ids.append(frame_id)
#         lbls.append(int(lbl))
#         x.append(int(float(gaze_x)))
#         y.append(int(float(gaze_y)))
#         print(frame_id, lbl, gaze_x, gaze_y)#dataline[6].split(',')[0], dataline[6].split(',')[1])

# train_lbl = np.asarray(lbls, dtype=np.int32)
# train_size = len(train_lbl)
# frame_ids = np.asarray(frame_ids)
# print(train_size)

# img1 = np.float32(preprocess(cv2.imread('img1.png')))
# # img2 = preprocess(cv2.imread('img2.png'))
# # img3 = preprocess(cv2.imread('img3.png'))
# # img4 = preprocess(cv2.imread('img4.png'))

# # input_img = np.dstack((img1, img2, img3, img4))

# img1=np.expand_dims(img1, axis=-1)

# print(img1.shape)

# input_img=np.expand_dims(img1, axis=0)

# print(input_img.dtype)

# gril = tf.keras.models.load_model('model_2.h5')

# action, gaze = gril.predict(input_img)
# print(action)
# print(np.argmax(action))

import glob

file = "/home/grads/m/mdsunbeam/atari_head/IL-CGL/data/alien"
for f in glob.glob(file + "/*.txt"):
    print(f[0:-3] + "tar.bz2")
