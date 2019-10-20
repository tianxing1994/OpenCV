"""
参考链接:
https://github.com/rmislam/PythonSIFT/blob/master/siftdetector.py
https://github.com/paulaner/SIFT
https://github.com/DLlearn/SIFT

SIFT 算法介绍
https://blog.csdn.net/u010440456/article/details/81483145
https://blog.csdn.net/lyl771857509/article/details/79675137

SIFT 算法:
1. 通过高斯差分法求图像中角点的位置.
(1) 对图像 I 进行不同程序的高斯模糊. 得到 g_1, g_2
(2) g_1 - g_2 得到高斯差分图像 d (相当于对图像求导, 求得每个像素的导数).
(3)
"""
import numpy as np
from scipy import signal
from scipy import misc
from scipy import ndimage
from scipy.stats import multivariate_normal
from numpy.linalg import norm
import numpy.linalg


def detect_keypoint(image, threshold):

    s = 3
    k = 2**(1.0/s)

    #
    kvec1 = np.array([1.3, 1.6, 1.6 * k, 1.6 * (k**2), 1.6 * (k**3), 1.6 * (k**4)])
    kvec2 = np.array([1.6 * (k**2), 1.6 * (k**3), 1.6 * (k**4), 1.6 * (k**5), 1.6 * (k**6), 1.6 * (k**7)])

    return
















































