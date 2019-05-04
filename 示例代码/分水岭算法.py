"""
相关函数:
cv.pyrMeanShiftFiltering
cv.getStructuringElement
cv.morphologyEx
cv.dilate
cv.distanceTransform
cv.normalize
cv.subtract
cv.connectedComponents
cv.watershed

路离变换


基于路离变换的分水岭分割流程
1. 输入图像
2. 灰度化
3. 二值化
4. 距离变换
5. 寻找种子
6. 生成 Marker
7. 分水岭变换
8. 输出图像
"""
import cv2 as cv
import numpy as np


def watershed_demo(image):
    print(image.shape)
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary-image", binary)

    # morphology operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(mb, kernel, iterations=3)
    cv.imshow('mor-opt', sure_bg)

    dist = cv.distanceTransform(mb, cv.DIST_L2, 3)
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow("distance-t", dist_output * 50)

    ret, surface = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)
    cv.imshow("surface-bin", surface)

    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(sure_bg)
    print(ret)


    # watershed transform
    markers = markers + 1
    markers[unknown==255] = 0
    markers = cv.watershed(image, markers=markers)
    image[markers==-1] = (0, 0, 255)
    cv.imshow("result", image)
    return


if __name__ == '__main__':
    src = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/coins.jpg")
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)

    watershed_demo(src)

    cv.waitKey(0)
    cv.destroyAllWindows()












