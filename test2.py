"""
参考链接:
https://blog.csdn.net/qq878594585/article/details/81901703
"""
# coding: utf-8
import numpy as np
import cv2

image_1 = cv2.imread(r'C:\Users\tianx\Desktop\splice_a.jpg')
image_2 = cv2.imread(r'C:\Users\tianx\Desktop\splice_b.jpg')

surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
kp1, des1 = surf.detectAndCompute(image=image_1, mask=None)
kp2, des2 = surf.detectAndCompute(image=image_2, mask=None)

indexParams = dict(algorithm=0, trees=5)
searchParams = dict(checks=50)

flann = cv2.FlannBasedMatcher(indexParams, searchParams)
matches = flann.knnMatch(queryDescriptors=des1,
                         trainDescriptors=des2,
                         k=2)

good = []

for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

src_pts = np.array([kp1[m.queryIdx].pt for m in good])
dst_pts = np.array([kp2[m.trainIdx].pt for m in good])

H = cv2.findHomography(src_pts, dst_pts)

print(len(good))
print(H)
print(type(H))
print(H[0].shape)
print(H[1].shape)
print(len(H))
