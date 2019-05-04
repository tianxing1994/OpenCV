"""
图像形态学:
是图像处理学科的一个单独分支学科
灰度与二值图像处理中重要手段
是由数学的集合论等相关理论发展起来的.

膨胀(Dilate):
用自定义长度的窗口扫描图片, 用窗口中的最大值作为窗口中心对应像素的值, 输出到新的图片中.

腐蚀(Erode):
用自定义长度的窗口扫描图片, 用窗口中的最小值作为窗口中心对应像素的值, 输出到新的图片中.
"""
import cv2 as cv
import numpy as np


def erode_demo(image):
    """
    腐蚀
    :param image:
    :return:
    """
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.erode(binary, kernel)
    cv.imshow("erode_demo", dst)
    return


def dilate_demo(image):
    """
    膨胀
    :param image:
    :return:
    """
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.dilate(binary, kernel)
    cv.imshow("erode_demo", dst)
    return


if __name__ == '__main__':
    src = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/contours.png")
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)

    erode_demo(src)
    # dilate_demo(src)

    cv.waitKey(0)
    cv.destroyAllWindows()