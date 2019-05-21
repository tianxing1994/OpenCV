"""
Canny 算法介绍：
1. 高斯模糊： GaussianBlur
2. 灰度转换: cvtColor
3. 计算梯度： Sobel/Scharr
4. 非最大信号抑制
5. 高低阈值输出二值图像
"""
import cv2 as cv
import numpy as np


def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    edge_output = cv.Canny(gray, 50, 150)

    # gradx = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    # grady = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    # edge_output = cv.Canny(gradx, grady, 50, 150)

    cv.imshow("Canny Edge", edge_output)

    # 彩色边缘
    # dst = cv.bitwise_and(image, image, mask=edge_output)
    # cv.imshow("Color Edge", dst)
    return


if __name__ == '__main__':
    src = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/lena.png")
    # cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    # cv.imshow("input image", src)

    edge_demo(src)

    cv.waitKey(0)
    cv.destroyAllWindows()