import cv2 as cv
import numpy as np

a = 0.88267063
b = 0.46999209

print(np.arctan2(a, b) * 180 / 3.1415 - 45)
