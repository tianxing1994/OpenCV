"""
参考链接:
https://blog.csdn.net/qq_38660394/article/details/80762011
相关函数:
cv2.inRange
"""
import cv2
import numpy as np

image_path = r'C:\Users\Administrator\PycharmProjects\openCV\dataset\other\wechat.jpg'
image = cv2.imread(image_path)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_red = np.array([30,150,50])
upper_red = np.array([255,255,180])
mask = cv2.inRange(hsv, lower_red, upper_red)
# bitwise_and, 用于位运算的两个数组完全一样, 则输出结果仍为 image, 但是由于指定了 mask, 则只有 mask 中为 0 的部分会输出黑色.
res = cv2.bitwise_and(image,image, mask= mask)
# res = np.array(image * (mask[:, :, np.newaxis] / 255), dtype=np.uint8)

cv2.imshow('image',image)
cv2.imshow('mask',mask)
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
