"""
相关函数:
cv2.threshold
cv2.adaptiveThreshold
"""
import cv2 as cv
import numpy as np


def threshold_demo(image):
    """
    cv2.THRESH_TRIANGLE: 适用于分割图像直方图只有单个峰值的图像.
    :param image:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # result, binary = cv.threshold(gray, 127, 255, cv.THRESH_TRIANGLE)
    result, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # result, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    # result, binary = cv.threshold(gray, 127, 255, cv.THRESH_MASK)
    # result, binary = cv.threshold(gray, 127, 255, cv.THRESH_OTSU)
    # result, binary = cv.threshold(gray, 127, 255, cv.THRESH_TOZERO)
    # result, binary = cv.threshold(gray, 127, 255, cv.THRESH_TOZERO_INV)
    # result, binary = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)

    print("threshold: ", result)
    cv.imshow("binary", binary)
    return


def local_threshold(image):
    """
    局部二值化: 用一个卷积核去扫描图像, 对卷积核内的像素求平均值, 当前值大于均值则为 1, 小于均值则为 0.
    cv.ADAPTIVE_THRESH_GAUSSIAN_C: 在卷积核内求均值时, 考虑每个像素的高斯权重.
    cv.ADAPTIVE_THRESH_MEAN_C: 在卷积核内求均值时, 每个像素具有相同的权重.
    :param image:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    result = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    # cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
    # print("threshold: ", result)
    cv.imshow("binary", result)
    return


def custom_threshold(image):
    """
    自定义: 基于整个图像的像素均值的二值化分割
    :param image:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w*h])
    mean = m.sum() / (w*h)
    print("mean: ", mean)
    result, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.imshow("binary", binary)
    return


def demo1():
    src = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/image0.JPG")
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)

    custom_threshold(src)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo2():
    """通过滑动条观察不同阈值下的二值图."""
    # image_path = '../dataset/data/other/silver.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572330104.jpg'
    image_path = '../dataset/local_dataset/snapshot_1572427571.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def callback(threshold):
        _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
        cv.imshow('image', binary)
        pass

    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.createTrackbar('threshold', 'image', 0, 255, callback)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return


if __name__ == '__main__':
    demo2()
