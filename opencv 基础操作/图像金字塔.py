"""
相关函数:
cv2.pyrDown
cv2.pyrUp
cv2.subtract
"""
import cv2 as cv
import numpy as np


def pyraimd_demo(image):
    level = 3
    template = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(template)
        # dst = cv.pyrUp(template)
        pyramid_images.append(dst)
        cv.imshow("pyramid_down_" + str(i), dst)
        template = dst.copy()
    return pyramid_images


def lapalian_demo(image):
    """
    要求图像大小应是 2^n 2 的 n 次方. 具体的我还没搞懂.
    :param image:
    :return:
    """
    pyramid_images = pyraimd_demo(image)
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("lapalian_down_" + str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expand)
            cv.imshow("lapalian_down_" + str(i), lpls)
    return


if __name__ == '__main__':
    src = cv.imread("../dataset/lena.png")


    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)

    pyraimd_demo(src)
    # lapalian_demo(src)

    cv.waitKey(0)
    cv.destroyAllWindows()