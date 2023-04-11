import numpy as np
import os
import random


def read_npz(data_path):
    with np.load(data_path) as data:
        l = len(data["images"])
        train_imgs = data['images']
        train_imgs = np.reshape(data['images'], (l, 224, 224, 1))
        #print(train_imgs.shape)
        train_gaze = data['heatmap']
        train_gaze = np.reshape(data['heatmap'], (l, 224, 224, 1))
        #print(train_gaze.shape)
        # train_act = data['vel_comm']
     #    print(train_act.shape)
        return train_imgs.shape, train_gaze.shape


def generate(path, file_list):
    # Generate batches of samples
    while 1:
        # indexes = folder_list
        imax = int(len(file_list)/1)  # 1,2,...number of npz files
        for i in range(imax):
            file_npz = [k for k in file_list[i*1:(i+1)*1]]
            # convert to string
            file_npz = ''.join(file_npz)
            # folder_number = folder_list[i]
            # Generate data
            # print folder_number
            X, y = read_npz(os.path.join(path, file_npz))

            yield X, y, file_npz


if __name__ == "__main__":
    # test the generator function
    datapath = "/scratch/user/ravikt/airsim/test_data/"
    file_list = os.listdir(datapath)
    random.shuffle(file_list)
    x = generate(datapath, file_list)
    print(next(x))
    print(next(x))
    print(next(x))
    print(next(x))
