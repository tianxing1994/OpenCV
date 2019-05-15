import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread(r'C:\Users\tianx\PycharmProjects\opencv\dataset\other\aa.jpg',0)
img2 = cv2.imread(r'C:\Users\tianx\PycharmProjects\opencv\dataset\other\bb.jpg',0)

orb = cv2.ORB_create(nfeatures=50)
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

print(len(kp1))

# NORM_L1 和 NORM_L2 是 SIFT 和 SURF 描述符的优先选择，NORM_HAMMING 和 NORM_HAMMING2 是用于 ORB 算法
# bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
matches = des2.match(queryDescriptors=des1)

print(matches)
print(len(matches))