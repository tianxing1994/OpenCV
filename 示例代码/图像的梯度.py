"""
相关函数:
cv2.Scharr
cv2.Sobel
cv2.convertScaleAbs
cv2.addWeighted
cv2.Laplacian
cv2.filter2D

Scharr 算子:
是 Sobel 算子的增强版本.

Sobel 算子:
水平梯度: Gx = [[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]
垂直梯度: Gy = [[-1, -2, -1], [0, 0, 0], [+1, +2, +1]]
最终图像梯度: G = √Gx2 + Gy2, G = |Gx| + |Gy|

laplian 算子:
[[0, 1, 0], [1, -4, 1], [0, 1, 0]]
[[1, 1, 1], [1, -8, 1], [1, 1, 1]]
"""
import cv2 as cv
import numpy as np


def sobel_demo(image):
    """
    求像素变化的一阶导数, 则图像边界处的值会为现最大值.
    :param image:
    :return:
    """
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)
    # grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    # grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    # cv.imshow("gradient-x", gradx)
    # cv.imshow("gradient-y", grady)
    cv.imshow("gradient-xy", gradxy)
    return


def lapalian_demo(image):
    """
    求像素变化的二阶导数, 则图像边界处的值为 0, 边界附近出现最大和最小值.
    :param image:
    :return:
    """
    dst = cv.Laplacian(image, cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("Laplian_demo", lpls)
    return


def custom_lapalian_demo(image):
    """
    利用算子, 自定义实现 cv2.Laplacian 方法
    :param image:
    :return:
    """
    # kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    dst = cv.filter2D(image, cv.CV_32F, kernel=kernel)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("custom_lapalian_demo", lpls)
    return


if __name__ == '__main__':
    src = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/image0.JPG")
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)

    custom_lapalian_demo(src)

    cv.waitKey(0)
    cv.destroyAllWindows()