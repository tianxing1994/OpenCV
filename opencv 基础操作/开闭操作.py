"""
相关函数:
cv.getStructuringElement
cv.morphologyEx
cv.threshold

开操作(Open):
图像形态学的重要操作之一, 基于膨胀与腐蚀操作组合形成的.
主要是应用在二值图像分析中, 灰度图像亦可.
开操作 = 腐蚀 + 膨胀, 输入图像 + 结构元素.
通过先腐蚀, 消除噪点; 再膨胀, 恢复图像.

闭操作(Close):
闭操作 = 膨胀 + 腐蚀, 输入图像 + 结构元素.
通过先膨胀, 将断点连接起来; 再腐蚀, 恢复图像.
"""
import cv2 as cv
import numpy as np


def open_demo1():
    """
    开操作: 消除噪点
    :param image:
    :return:
    """
    image = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/morph.png")
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow("open-result", binary)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return

def open_demo2():
    """
    开操作: 消除竖直线
    :param image:
    :return:
    """
    image = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/morph01.png")
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow("open-result", binary)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def close_demo(image):
    """
    闭操作
    :param image:
    :return:
    """
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cv.imshow("close-result", binary)
    return


if __name__ == '__main__':
    open_demo2()


