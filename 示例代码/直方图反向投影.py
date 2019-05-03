"""
相关函数:
cv2.calcHist
cv2.normalize
cv2.calcBackProject
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def back_projection_demo():
    sample = cv.imread('C:/Users/tianx/PycharmProjects/opencv/dataset/example.png')
    target = cv.imread('C:/Users/tianx/PycharmProjects/opencv/dataset/target.jpg')
    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    cv.imshow("sample", sample)
    cv.imshow("target", target)

    roiHist = cv.calcHist([roi_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject(target, [0, 1], roiHist, [0, 180, 0, 256], 1)
    cv.imshow('backProjectionDemo', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def hist2d_demo(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([image], [0, 1], None, [180, 256], [0, 180, 0, 256])

    plt.imshow(hist, interpolation='nearest')
    plt.title('2D Histogram')
    plt.show()
    return


if __name__ == '__main__':
    # src = cv.imread('C:/Users/tianx/PycharmProjects/opencv/dataset/example.png')
    # cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    # cv.imshow("input image", src)
    # hist2d_demo(src)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    back_projection_demo()














