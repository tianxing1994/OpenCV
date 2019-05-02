"""
相关函数:
cv2.calcHist
cv2.equalizeHist
cv2.createCLAHE
cv2.apply
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()
    return


def image_hist(image):
    """
    分别显示 BGR 三个通道的直方图
    :param image:
    :return:
    """
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()
    return


def equalHist_demo(image):
    """
    直方图均衡化 1:
    :param image:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    result = cv.equalizeHist(gray)
    return result


def clahe_demo(image):
    """
    直方图均衡化 2:
    :param image:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    result = clahe.apply(gray)
    return result


if __name__ == '__main__':
    """
    image_path = 'C:/Users/tianx/PycharmProjects/opencv/dataset/example.png'
    src = cv.imread(image_path)
    plot_demo(src)
    """

    image_path = 'C:/Users/tianx/PycharmProjects/opencv/dataset/rice.png'
    src = cv.imread(image_path)
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)

    src = clahe_demo(src)

    cv.imshow("input image", src)
    cv.waitKey(0)
    cv.destroyAllWindows()





