import cv2 as cv
import numpy as np


nd1 = np.array([[1820.,  440.], [66.,  397.], [92., 1158.], [1808., 1146.]], dtype=np.float32)
nd2 = np.array([[1440, 0], [0, 0], [0, 1920], [1440, 1920]], dtype=np.float32)


perspective_matrix = cv.getPerspectiveTransform(src=nd1, dst=nd2)

print(perspective_matrix)