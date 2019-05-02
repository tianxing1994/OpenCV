"""
相关函数:
cv2.split
cv2.merge
"""
import cv2 as cv
import numpy as np


def extrace_object_demo():
    """
    课件中的视频我没有.
    :param image:
    :return:
    """
    capture = cv.VideoCapture(r"C:\Users\tianx\PycharmProjects\opencv\dataset\data\768x576.avi")
    while True:
        ret, frame = capture.read()
        if ret == False:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([37, 43, 46])
        upper_hsv = np.array([77, 255, 255])
        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        cv.imshow("video", frame)
        cv.imshow("mask", mask)
        c = cv.waitKey(40)
        if c == 27:
            break


def color_space_demo(image):
    """
    将 RGB 图像转换到 GRAY, HSV, YUV, YCrCb 四种色彩空间.
    :param image:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    cv.imshow("ycrcb", ycrcb)


if __name__ == '__main__':
    image_path = 'C:/Users/tianx/PycharmProjects/opencv/dataset/lena.png'
    src = cv.imread(image_path)
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)
    color_space_demo(src)
    cv.waitKey(0)
    cv.destroyAllWindows()
























