import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


image = np.zeros([400, 400, 3], np.uint8)
image[100:300, 100:300, :] = 255
cv.imshow("fill_binary", image)

mask = np.ones([402, 402, 1], np.uint8)
mask[101:301, 101:301] = 0
cv.floodFill(image=image,
             mask=mask,
             seedPoint=(200, 200),
             newVal=(255, 0, 0),
             upDiff=cv.FLOODFILL_MASK_ONLY)
cv.imshow("filled_binary", image)
cv.waitKey(0)
cv.destroyAllWindows()