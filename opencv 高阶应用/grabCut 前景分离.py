"""
参考链接:
https://blog.csdn.net/zouxy09/article/details/8534954
https://docs.opencv.org/3.4.3/d8/d83/tutorial_py_grabcut.html

官方 python 演示:
https://github.com/opencv/opencv/blob/master/samples/python/grabcut.py
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

image_path = r'C:\Users\Administrator\PycharmProjects\openCV\dataset\other\grabCut.jpg'
image = cv2.imread(image_path)
mask = np.zeros(image.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (161,79,150,150)

cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = image*mask2[:,:,np.newaxis]
plt.imshow(img)
plt.colorbar()
plt.show()