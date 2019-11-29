import cv2 as cv
import numpy as np


points = np.array([[[10, 10]],
                   [[20, 20]],
                   [[10, 30]]])
result = cv.minEnclosingCircle(points)
print(result)