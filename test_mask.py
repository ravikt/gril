import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def get_mask(center, size, sig):
    y, x = np.ogrid[-center[0]: size[0] -
                    center[0], -center[1]: size[1] - center[1]]
    print(x, y)
    keep = x * x + y * y < 1

    # keep = x * x + y * y < h**2 + w**2
    # plt.imshow(keep)
    # plt.show()
    mask = np.zeros(size)
    # print(keep)
    mask[keep] = 1
    # print(np.int_(np.floor(center[0])), np.int_(np.floor(center[1])))
    # mask[np.floor(center[0]), np.floor(center[1])] = 1
    mask = gaussian_filter(mask, sigma=sig)
    print(mask)
    print(mask.sum())

    return mask / mask.sum()


h, w = 400, 700
# gaze_x, gaze_y = 0.50418, 0.558429
gaze_x, gaze_y = 0.765101, 0.537377
dim_x, dim_y = h, w

mask = get_mask([gaze_y*h, gaze_x*w], [dim_x, dim_y], 15)
# mask = get_mask([gaze_x*w, gaze_y*h], [h, w], 15)


print(mask.shape)
print(np.isnan(mask))
plt.imshow(mask)
plt.show()
