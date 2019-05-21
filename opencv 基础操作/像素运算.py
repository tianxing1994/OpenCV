"""
相关函数:
cv2.add
cv2.subtract
cv2.divide
cv2.multiply
cv2.meanStdDev
cv2.mean
cv2.bitwise_and
cv2.bitwise_or
cv2.bitwise_not
cv2.bitwise_xor
"""
import cv2 as cv
import numpy as np


def add_demo(m1, m2):
    dst = cv.add(m1, m2)
    return dst


def subtract_demo(m1, m2):
    dst = cv.subtract(m1, m2)
    return dst


def divide_demo(m1, m2):
    dst = cv.divide(m1, m2)
    return dst


def multiply_demo(m1, m2):
    dst = cv.multiply(m1, m2)
    return dst


def others(m1, m2):
    M1 = cv.mean(m1)
    M2 = cv.mean(m2)
    print(M1)
    print(M2)
    return m1


def meanStdDev(m1, m2):
    M1, dev1 = cv.meanStdDev(m1)
    M2, dev2 = cv.meanStdDev(m2)
    print(M1)
    print(M2)

    print(dev1)
    print(dev2)
    return m1


def logic_demo(m1, m2, method='and'):
    """
    按位运算, and, or, not, xor.
    :param m1:
    :param m2:
    :param method:
    :return:
    """
    if method=='and':
        dst = cv.bitwise_and(m1, m2)
    elif method=='or':
        dst = cv.bitwise_or(m1, m2)
    elif method=='not':
        dst = cv.bitwise_not(m1, m2)
    elif method=='xor':
        dst = cv.bitwise_xor(m1, m2)
    else:
        raise ValueError('The method %s is invalid' % method)
    return dst


def contrast_brightness_demo(image, c=1, b=0):
    """
    通过 addWeighted 方法, 调整数像的对比度,亮度.
    :param image:
    :param c: 对比度, 取 1 时, 图像不变. 通过将图像中的每一个像素乘以一个系数来实现,
    即增大了像素值之间的距离, 超过 255 的当作 255 来处理. 因此会有亮度被提升的感觉.
    :param b: 亮度, 取 0 时, 亮度不变. 提升对比度的同时, 将亮度设置为一个负值,
    使得图像变得很鲜艳, 但又不那么得亮.
    :return: 被调整对比度, 亮度之后的图像.
    """
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    return dst


if __name__ == '__main__':
    image_path1 = 'C:/Users/tianx/PycharmProjects/opencv/dataset/data/LinuxLogo.jpg'
    image_path2 = 'C:/Users/tianx/PycharmProjects/opencv/dataset/data/WindowsLogo.jpg'
    image_path3 = 'C:/Users/tianx/PycharmProjects/opencv/dataset/lena.png'

    src1 = cv.imread(image_path1)
    src2 = cv.imread(image_path2)
    src3 = cv.imread(image_path3)
    cv.namedWindow("input image0", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image0", src3)

    img = contrast_brightness_demo(src3, 2, -100)

    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()