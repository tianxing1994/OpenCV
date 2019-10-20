import cv2 as cv
import numpy as np


image_path = 'dataset/data/image_sample/lena.png'
image = cv.imread(image_path)

result = cv.selectROI(windowName='selectROI', img=image, showCrosshair=False, fromCenter=False)
print(result)

