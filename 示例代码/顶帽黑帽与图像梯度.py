"""
顶帽(tophat):
原图像与开操作之间的差值图像

黑帽(blackhat):
原图像与闭操作之间的差值图像

形态学梯度(Gradient):
基本梯度: 是用膨胀后的图像减去腐蚀后的图像得到差值图像, 称为梯度图像,
也是 opencv 中支持的计算形态学梯度的方法, 而此方法得到梯度又被称为基本梯度.

内部梯度: 是用原图像减去腐蚀之后的图像得到差值图像, 称为图像的内部梯度.

外部梯度: 图像膨胀之后再减去原来的图像得到的差值图像, 称为图像的外部梯度.
"""
import cv2 as cv
import numpy as np


def top_hat_demo(image):
    """
    顶帽
    :param image:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    cv.imshow("tophat", dst)
    return


def black_hat_demo(image):
    """
    黑帽
    :param image:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    cv.imshow("blackhat", dst)
    return


def hat_gray_demo(image):
    """
    应用于二值图像
    :param image:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # dst = cv.morphologyEx(binary, cv.MORPH_TOPHAT, kernel)
    dst = cv.morphologyEx(binary, cv.MORPH_BLACKHAT, kernel)
    cv.imshow("blackhat", dst)
    return


def image_gradient(image):
    """
    内外梯度:
    :param image:
    :return:
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dm = cv.dilate(image, kernel)
    em = cv.erode(image, kernel)

    dst1 = cv.subtract(image, em)
    dst2 = cv.subtract(dm, image)

    cv.imshow("internal", dst1)
    cv.imshow("external", dst2)
    return



if __name__ == '__main__':
    src = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/lena.png")
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)

    image_gradient(src)

    cv.waitKey(0)
    cv.destroyAllWindows()
















