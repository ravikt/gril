'''
Example data loader for Atari-HEAD dataset
This file reads dataset by
Zhang, R., Walshe, C., Liu, Z., Guan, L., Muller, K., Whritner, J., ... & Ballard, D. (2020, April). Atari-head: Atari human eye-tracking and demonstration dataset. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 04, pp. 6811-6820).
'''

import sys, os, re, threading, time, copy
import numpy as np
import tarfile
import cv2


def preprocess(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width = 84
    height = 84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame / 255.0

class Dataset:
  def __init__(self, tar_fname, label_fname):
    t1=time.time()
    print ("Reading all training data into memory...")

    # Read action labels from txt file
    frame_ids, lbls = [], [] 
    with open(label_fname,'r') as f:
        for line in f:
            if line.startswith("frame_id") or line == "": 
                continue # skip head or empty lines
            dataline = line.split(',') 
            frame_id, lbl = dataline[0], dataline[5]
            if lbl == "null": # end of file
                break
            frame_ids.append(frame_id)
            lbls.append(int(lbl))
    self.train_lbl = np.asarray(lbls, dtype=np.int32)
    self.train_size = len(self.train_lbl)
    self.frame_ids = np.asarray(frame_ids)
    print(self.train_size)

    # Read training images from tar file
    imgs = [None] * self.train_size
    print("Making a temp dir and uncompressing PNG tar file")
    temp_extract_dir = "img_data_tmp/"
    if not os.path.exists(temp_extract_dir):
        os.mkdir(temp_extract_dir)
    tar = tarfile.open(tar_fname, 'r')
    tar.extractall(temp_extract_dir)
    png_files = tar.getnames()
    # get the full path
    temp_extract_full_path_dir = temp_extract_dir + png_files[0].split('/')[0]
    print("Uncompressed PNG tar file into temporary directory: " + temp_extract_full_path_dir)

    print("Reading images...")
    for i in range(self.train_size):
        frame_id = self.frame_ids[i]
        png_fname = temp_extract_full_path_dir + '/' + frame_id + '.png'
        img = np.float32(cv2.imread(png_fname))
        img = preprocess(img)
        imgs[i] = copy.deepcopy(img)
        print("\r%d/%d" % (i+1,self.train_size)),
        sys.stdout.flush()

    self.train_imgs = np.asarray(imgs)
    print ("Time spent to read training data: %.1fs" % (time.time()-t1))

  def standardize(self):
    self.mean = np.mean(self.train_imgs, axis=(0,1,2))
    self.train_imgs -= self.mean # done in-place --- "x-=mean" is faster than "x=x-mean"

  def load_predicted_gaze_heatmap(self, train_npz):
    train_npz = np.load(train_npz)
    self.train_GHmap = train_npz['heatmap']
    # npz file from pastK models has pastK-fewer data, so we need to know use value of pastK
    pastK = 4
    self.train_imgs = d.train_imgs[pastK:]
    self.train_lbl = d.train_lbl[pastK:]

  def reshape_heatmap_for_cgl(self, heatmap_shape):
    # predicted human gaze was in 84 x 84, needs to be reshaped for cgl
    #heatmap_shape: output feature map size of the conv layer 
    import cv2
    for i in range(len(self.train_GHmap)):
        self.train_GHmap[i] = cv2.resize(frame, (heatmap_shape, heatmap_shape), interpolation=cv2.INTER_AREA)

  def generate_data_for_gaze_prediction(self):
    self.gaze_imgs = [None] * (self.train_size - 3)
    #stack every four frames to make an observation (84,84,4)
    for i in range(3, self.train_size):
        stacked_obs = np.zeros((84, 84, 4))
        stacked_obs[:, :, 0] = self.train_imgs[i-3]
        stacked_obs[:, :, 1] = self.train_imgs[i-2]
        stacked_obs[:, :, 2] = self.train_imgs[i-1]
        stacked_obs[:, :, 3] = self.train_imgs[i]
        self.gaze_imgs[i-3] = copy.deepcopy(stacked_obs)

    self.gaze_imgs = np.asarray(self.gaze_imgs)
    return self.gaze_imgs
    print("Shape of the data for gaze prediction: ", self.gaze_imgs.shape)

if __name__ == "__main__":
    d = Dataset(sys.argv[1], sys.argv[2]) #tarfile (images), txtfile (labels)
    # For gaze prediction
    d.generate_data_for_gaze_prediction()
    # For training imitation learning algorithms (cgl, agil)
    d.load_predicted_gaze_heatmap(sys.argv[3]) #npz file (predicted gaze heatmap)
    d.standardize() #for training imitation learning only, gaze model has its own mean files
