import numpy as np
import cv2


image = cv2.imread('pos-129.pgm')

cv2.imshow('image',image)
cv2.waitKey()
cv2.destroyAllWindows()