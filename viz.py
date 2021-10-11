import numpy as np
import cv2
import matplotlib.pyplot as plt

dfile = "results.npz"
images = np.load(dfile)["heatmap"]
print(images.shape)


plt.figure()
plt.imshow(images[0])
plt.savefig('sample.png')
