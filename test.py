import cv2 as cv
import numpy as np


nd1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
nd2 = np.reshape(nd1, newshape=(2, 4, 2))
print(nd2)
