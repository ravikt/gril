import csv
import cv2
import copy
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter
import torch


def reshape_image(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width = 84
    height = 84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame / 255.0


def reshape_heatmap(heatmap):
    ghmap = []
    # print(heatmap.shape, type(heatmap))
    heatmap = heatmap.numpy()
    # print(heatmap[0].shape, len(heatmap))

    for i in range(len(heatmap)):
        ghmap.append(cv2.resize(
            heatmap[i], (84, 84), interpolation=cv2.INTER_AREA))

    return np.array(ghmap)


def get_mask(center, size, sig):
    y, x = np.ogrid[-center[0]: size[0] -
                    center[0], -center[1]: size[1] - center[1]]
    keep = x * x + y * y < 1
    mask = np.zeros(size)
    mask[keep] = 1
    mask = gaussian_filter(mask, sigma=sig)
    print(mask.sum())

    return mask / mask.sum()


h = 480  # row
w = 704  # column
# gaze_pos = np.array(gaze_pos)
# gaze_pos = gaze_pos.astype(float)

# plt.figure()
# plt.imshow(out)
# plt.show()


def preprocess_gaze_heatmap(gaze_2ds, sigma):
    ''' Convert 1-hot gaze heatmap to gaussian distribution; 
    note that this assumes each frame only has 0 or 1 valid gaze positions'''
    gmaps = np.zeros([len(gaze_2ds), h, w, 1], dtype=np.float32)
    dim_x, dim_y = h, w  # gmaps.shape[1], gmaps.shape[2]
    # x and y not necessarily means values along x and y axis
    # dim_y, dim_x = h, w#gmaps.shape[1], gmaps.shape[2]

    print(gaze_2ds.shape)
    for i in range(len(gaze_2ds)):
        gaze_x, gaze_y = gaze_2ds[i][0], gaze_2ds[i][1]
        if gaze_x == -1:
            mask = np.ones((dim_x, dim_y)) / (dim_x * dim_y)  # uniform
        else:
            # mask = get_mask([gaze_x*w, gaze_y*h], [dim_x, dim_y], sigma)
            mask = get_mask([gaze_y*h, gaze_x*w], [dim_x, dim_y], sigma)

        mask = np.expand_dims(mask, axis=2)
        gmaps[i] = copy.deepcopy(mask)
        # print(gmaps[i])
        print(np.any(np.isnan(gmaps[i])))
        plt.imshow(np.squeeze(gmaps[i], axis=2))
        plt.imsave(f'out{i}.png', np.squeeze(gmaps[i], axis=2))

    gmaps = torch.tensor(gmaps, dtype=torch.float32).permute(0, 3, 1, 2)
    return gmaps


if __name__ == "__main__":

    gaze_pos = []

    with open('log.csv', 'r') as csvfile:
        gaze = csv.reader(csvfile)
        for row in islice(gaze, 930, None):
            # print(row[-3], row[-2])
            coords = np.hstack((row[-3], row[-2]))
            gaze_pos.append(coords)

    print(gaze_pos[0])
    print(gaze_pos[0].shape)
    x, y = gaze_pos[0]
    print(x, y)
    output = preprocess_gaze_heatmap(np.array(gaze_pos), 3)
    print(output.shape)
