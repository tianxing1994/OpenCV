"""
相关函数:
cv2.findContours
cv2.drawContours
轮廓发现:
是基于图像边缘提取的基础寻找对象轮廓的方法.
所以边缘提取的阈值选定会影响最终轮廓发现结果.
"""
import cv2 as cv
import numpy as np


def countours_demo(image):
    dst = cv.GaussianBlur(image, (5, 5), 0)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary image", binary)

    contours, heriachy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 2)
        # cv.drawContours(image, contours, i, (0, 0, 255), -1)    # 填充轮廓
        print(i)
    cv.imshow("detect_contours", image)
    return


def canny_countours_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    edge_output = cv.Canny(gray, 50, 150)
    # gradx = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    # grady = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    # edge_output = cv.Canny(gradx, grady, 50, 150)
    cv.imshow("Canny Edge", edge_output)

    contours, heriachy = cv.findContours(edge_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 2)
        # cv.drawContours(image, contours, i, (0, 0, 255), -1)    # 填充轮廓
        print(i)
    cv.imshow("detect_contours", image)
    return


if __name__ == '__main__':
    src = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/coins.jpg")
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)

    # countours_demo(src)
    canny_countours_demo(src)

    cv.waitKey(0)
    cv.destroyAllWindows()