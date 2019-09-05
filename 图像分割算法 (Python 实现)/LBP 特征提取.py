"""
参考链接:
https://blog.csdn.net/u010006643/article/details/46417091

LBP 算法

圆形 LBP 算子
如:
[[1, 2, 2],
 [9, 5, 6],
 [5, 3, 1]]

[[0, 0, 0],
 [1,  , 1],
 [1, 0, 0]]

 binary = 00010011
 decimal = 19

该算子将像素八邻域的像素值大于中心点值的标注为 1, 反之标注为 0. 这样我们就会得到一个八位数. 这时从左上角开始算最高位,
最后左边的为最低位, 在这个图中就是 00010011, 换算十进制就是 19, 我们就把这个值作为中心点的 LBP 算子值.

为了让 LBP 特值取有旋转不变性, 将二进制串进行旋转. 假设一开始得到的 LBP 特征为 10010000,
那么将这个二进制特征按照顺时针方向旋转, 可以转化为 00001001 的形式, 这样得到的 LBP 值是最小的.
无论图像怎么旋转, 对点提取的二进制特征的最小值是不变的. 用最小值作为提取的 LBP 特征, 这样 LBP 就是旋转不变的.
"""
import os
import math
import numpy as np
import cv2 as cv


def local_binary_pattern(image, r=2, p=8):
    """
    提取图像的 LBP 特征.
    :param image: 输入灰度图像
    :param r: 指定 LBP 特征提取窗口半径大小.
    :param p:
    :return:
    """
    # LBP 算子 8 邻域像素相对位置索引. 左上角为起点.
    region8_i = [-1, 0, 1, 1, 1, 0, -1, -1]
    region8_j = [-1, -1, -1, 0, 1, 1, 1, 0]
    result = np.zeros(shape=image.shape)

    h, w = image.shape
    for i in range(r, h-r):
        for j in range(r, w-r):
            binary_lbp = ''
            center_pixel = image[i, j]
            for relative_i, relative_j in zip(region8_i, region8_j):
                i_ = i + relative_i
                j_ = j + relative_j
                if image[i_, j_] > center_pixel:
                    binary_lbp += '1'
                else:
                    binary_lbp += '0'
            result[i, j] = int(binary_lbp.strip('0'), base=2)
    return result


















