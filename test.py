import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = np.zeros([40, 40, 1], np.uint8)

mask = np.ones([42, 42, 1], np.uint8)
mask[11:31, 11:31] = 0

retval, image, mask, rect = cv.floodFill(image=image,
                                         mask=mask,
                                         seedPoint=(20, 20),
                                         newVal=0.5)

plt.imshow(image.reshape(40,40), cmap='gray')
plt.show()
plt.imshow(mask.reshape(42,42), cmap='gray')
plt.show()
