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


def show_image(image):
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


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


def gray_hist(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # images, channels, mask, histSize, ranges, hist=None, accumulate=None
    # histSize: 直方图的 bin 的数量.
    # ranges: 用于计算直方图的值的范围, 范围外的值不算入.
    hist = cv.calcHist(images=[gray], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    print(hist)
    print(hist.shape)
    print(hist.max())
    print(hist.T.shape)
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
    image_path = r'C:\Users\Administrator\PycharmProjects\OpenCV\dataset\image0.JPG'
    image = cv.imread(image_path)

    gray_hist(image)
