import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

dfile = "/scratch/user/ravikt/small.npz"
data  = np.load(dfile)
heatmap = data["heatmap"]
images  = data["images"]

# Converts the images to valid input
# for plt imshow
# images = np.array(images,np.int32)

# Convert heatmap from Nx1xMxN
# to NxMxN
# heatmap = np.squeeze(heatmap, axis=1)

print(heatmap.shape)
