"""
相关函数:
cv2.bilateralFilter
cv2.pyrMeanShiftFiltering
"""
import cv2 as cv
import numpy as np


def bi_demo(image):
    """
    高斯双边模糊: 高斯模糊的变种, 当卷积核内存在与当前值相差很大的值时,
    直接舍弃这些值, 而后计算加权平均.
    该方法在模糊图像的同时, 可以保留(不会模糊)图像的边缘, 可用作美颜
    :param image:
    :return:
    """
    result = cv.bilateralFilter(image, 0, 100, 15)
    return result


def shift_demo(image):
    """
    均值迁移模糊: 猜测:
    1. 与当前值相差较大的值会被舍弃.
    2. 计算当前卷积核内的剩余值质心, 卷积核的中心移动至计算出的质心.
    3. 考虑第 1 步, 重新计算质心, 移动卷积核.
    4. 迭代 1, 2, 3 步, 直至当前值与当前质心满足一定条件, 或迭代次数满足一定条件.
    https://blog.csdn.net/tengfei461807914/article/details/80412482
    :param image:
    :return:
    """
    result = cv.pyrMeanShiftFiltering(image, 10, 50)
    return result


if __name__ == '__main__':
    image_path = 'C:/Users/tianx/PycharmProjects/opencv/dataset/example.png'
    src = cv.imread(image_path)
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)

    src = shift_demo(src)

    cv.imshow("input image1", src)
    cv.waitKey(0)
    cv.destroyAllWindows()