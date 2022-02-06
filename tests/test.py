'''
Test script to assert npz data
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

#dfile = "/scratch/user/ravikt/sample/train_data/small.npz"
#dfile = "../utils/Fly_to_the_nearest_truck.npz"
d = "/scratch/user/ravikt/airsim/test_data/truck_mountains3.npz"
data  = np.load(d)
heatmap = data["heatmap"]
images  = data["images"]
#action = data["action"]
# Converts the images to valid input
# for plt imshow
# images = np.array(images,np.int32)

# Convert heatmap from Nx1xMxN
# to NxMxN
# heatmap = np.squeeze(heatmap, axis=1)

print(heatmap.shape)
print(images.shape)
