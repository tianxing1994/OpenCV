import cv2
import numpy as np
from scipy import ndimage

image_path = r'C:\Users\Administrator\PycharmProjects\openCV\dataset\other\sanshui.jpg'

# flags=0, 则读出的图像为灰度图.
image = cv2.imread(image_path, flags=0)

kernel_3 = np.array([[-1, -1, -1],
                     [-1, 8, -1],
                     [-1, -1, -1]])

kernel_5 = np.array([[-1, -1, -1, -1, -1],
                     [-1, 1, 2, 1, -1],
                     [-1, 2, 4, 2, -1],
                     [-1, 1, 2, 1, -1],
                     [-1, -1, -1, -1, -1]])

k3 = ndimage.convolve(image, kernel_3)
k5 = ndimage.convolve(image, kernel_5)

blurred = cv2.GaussianBlur(image, (11, 11), 0)

g_hpf = image - blurred

cv2.imshow('k3', k3)
cv2.imshow('k5', k5)
cv2.imshow('g_hpf', g_hpf)
cv2.waitKey(0)
cv2.destroyAllWindows()



