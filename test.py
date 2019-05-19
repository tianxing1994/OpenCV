import cv2
import numpy as np
from scipy import ndimage

image_path = r'C:\Users\Administrator\PycharmProjects\openCV\dataset\other\image.jpg'
image = cv2.imread(image_path, 0)

_, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[0]

result = cv2.arcLength(contour, True)
print(result)