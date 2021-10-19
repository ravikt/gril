import cv2
import glob
import numpy as np 
import os


a = cv2.imread('data/rgb_0.png')
a1 = cv2.resize(a, (84, 84), interpolation= cv2.INTER_LINEAR)

print(a.shape)

cv2.imwrite('out.png', a1)
#cv2.imshow('output',a1)
#cv2.waitKey()
#cv2.destroyAllWindows()

