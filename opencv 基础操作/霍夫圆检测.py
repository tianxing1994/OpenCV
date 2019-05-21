"""
相关函数:
cv2.pyrMeanShiftFiltering
cv2.HoughCircles

因为霍夫圆检测对噪声比较敏感, 所以首先要对图像做中值滤波.
基于效率考虑, Opencv 中实现的霍夫变换圆检测是基于图像梯度的实现, 分为两步:
1. 检测边缘, 发现可能的圆心
2. 基于第一步的基础上从候选圆心开始计算最佳半径大小
"""

import cv2 as cv
import numpy as np


def detect_circles_demo(image):
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)
    cimage = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimage, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
    cv.imshow("circles", image)
    return


if __name__ == '__main__':
    src = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/coins.jpg")
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)

    detect_circles_demo(src)

    cv.waitKey(0)
    cv.destroyAllWindows()
