import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

dfile = "car.npz"
data  = np.load(dfile)
heatmap = data["heatmap"]
images  = data["images"]

# Converts the images to valid input
# for plt imshow
# images = np.array(images,np.int32)

# Convert heatmap from Nx1xMxN
# to NxMxN
# heatmap = np.squeeze(heatmap, axis=1)

x = np.argwhere(np.isnan(heatmap))

print(x.shape)
# plt.figure()
# plt.imshow(images[1])
plt.imshow(heatmap[2])

plt.show()