import numpy as np
import cv2
import matplotlib.pyplot as plt

dfile = np.load("small_tm3.npz")
print(dfile.keys())
ghmap = dfile["heatmap"]
img   = dfile["images"]

#print(images.shape)


idx = [4, 55, 100, 400]
#plt.figure()

for i in idx:
   plt.imshow(img[i])
   plt.imsave(f'img_{i}.png', img[i])

   plt.imshow(ghmap[i])
   plt.imsave(f'gh_{i}.png', ghmap[i])
