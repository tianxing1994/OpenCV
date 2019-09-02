"""
相关函数:
cv2.calcHist
cv2.normalize
cv2.calcBackProject
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def back_projection_demo_ok():
    """
    直方图反向投影.
    :return:
    """
    sample = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/other/sample.png")
    target = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/other/target.png")
    roi_hsv = cv.cvtColor(sample,cv.COLOR_BGR2HSV)
    tar_hsv = cv.cvtColor(target,cv.COLOR_BGR2HSV)
    print(roi_hsv.shape)
    print(tar_hsv.shape)

    cv.imshow("sample",sample)
    cv.imshow("target",target)

    roihist = cv.calcHist([roi_hsv], [0, 1], None, [324, 356], [0, 324, 0, 356])    #加红部分越小，匹配越放松，匹配越全面，若是bsize值越大，则要求得越精细，越不易匹配，所以导致匹配出来的比较小
    cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)  #规划到0-255之间
    dst = cv.calcBackProject([tar_hsv],[0,1],roihist,[0,324,0,356],1)   #直方图反向投影
    print(dst.shape)
    cv.imshow("back_projection_demo",dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


def back_projection_demo():
    sample = cv.imread('C:/Users/tianx/PycharmProjects/opencv/dataset/example.png')
    target = cv.imread('C:/Users/tianx/PycharmProjects/opencv/dataset/target.jpg')

    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)
    print("roi_hsv: ", roi_hsv.shape)
    # cv.imshow("sample", sample)
    # cv.imshow("target", target)

    roiHist = cv.calcHist([target_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)
    print("直方图: ", roiHist.shape)

    dst = cv.calcBackProject(roi_hsv, [0, 1], roiHist, [0, 180, 0, 256], 1)
    print(dst.shape)
    cv.imshow('backProjectionDemo', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def hsv_image_back_project():
    # 彩色直方图反向投影
    image = cv.imread(r'C:\Users\Administrator\PycharmProjects\HuaWaQiangDan\code\CFR2.jpg')
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    roihist = cv.calcHist([image], [0, 1], None, [324, 356], [0, 324, 0, 356])
    cv.normalize(roihist, roihist, 0, 255, cv.NORM_MINMAX)
    result = cv.calcBackProject([image], [0, 1], roihist, [0, 324, 0, 356], 1)
    return result


def gray_image_back_project(image):
    # 灰度直方图反向投影
    roihist = cv.calcHist([image], [0], None, [255], [0, 255])
    cv.normalize(roihist, roihist, 0, 255, cv.NORM_MINMAX)
    result = cv.calcBackProject([image], [0], roihist, [0, 255], 1)
    return result


if __name__ == '__main__':
    # src = cv.imread('C:/Users/tianx/PycharmProjects/opencv/dataset/example.png')
    # cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    # cv.imshow("input image", src)
    # hist2d_demo(src)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    back_projection_demo_ok()














