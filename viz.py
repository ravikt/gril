import numpy as np
import cv2
import matplotlib.pyplot as plt

dfile = np.load("results.npz")
print(dfile.keys())
ghmap = dfile["heatmap"]
#img   = dfile["images"]

#print(images.shape)


idx = [0, 44, 1000, 5000]
#plt.figure()

for i in idx:
   #plt.imshow(img[i])
   #plt.savefig(f'img_{i}.png')

   plt.imshow(ghmap[i])
   plt.imsave(f'gh_{i}.png', ghmap[i])
