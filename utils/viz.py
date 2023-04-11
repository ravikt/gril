import numpy as np
import cv2
import matplotlib.pyplot as plt


#dfile = np.load("/scratch/user/ravikt/airsim/truck_mount_train/moving_truck_mountains1.npz")

dfile = np.load("truck_mountains1.npz")
print(dfile.keys())
ghmap = dfile["heatmap"]
img   = dfile["images"]

print(img.shape)
print(ghmap.shape)


idx = [33, 140]
#plt.figure()

for i in idx:
   plt.axis('off')
   plt.imshow(img[i], cmap='gray')
   plt.savefig(f'img_{i}.png', transparent=True, bbox_inches='tight')
   #print(img[i].shape)
   #plt.imsave(f'img_{i}.png', img[i], cmap="hot")
   #cv2.imwrite(f'img_{i}.png', img[i]*255)

   plt.imshow(ghmap[i])
   plt.savefig(f'gh_{i}.png', transparent=True, bbox_inches='tight')
   #print(ghmap[i].shape)
   #plt.imsave(f'gh_{i}.png', ghmap[i], cmap="hot") 
   #cv2.imwrite(f'gh_{i}.png', ghmap[i]*255)
