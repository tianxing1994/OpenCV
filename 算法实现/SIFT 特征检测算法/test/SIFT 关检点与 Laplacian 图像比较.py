import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return


def demo1():
    """显示拉普拉斯滤波后的图像"""
    image_path = "../../../dataset/data/image_sample/lena.png"
    image = cv.imread(image_path)
    image = cv.Laplacian(src=image, ddepth=cv.CV_8U, ksize=1)
    show_image(image, "Laplacian")
    return


def demo2():
    image_path = '../../../dataset/data/image_sample/lena.png'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    sift = cv.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    for keypoint in keypoints:
        x, y = keypoint.pt
        cv.circle(image, (int(x), int(y)), 1, (0, 0, 255), 1, cv.LINE_AA)

    show_image(image, "SIFT")
    return


def demo3():
    """分析: 拉普拉斯滤波后的图像显示出很多斑点, SIFT 算法把这些斑点作为特征点. """
    demo1()
    demo2()
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


if __name__ == '__main__':
    demo3()
