"""
相关函数:
cv2.calcHist
cv2.normalize
cv2.calcBackProject
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    """
    基于直方图反向投影的目标检测.
    sample.png 是目标的颜色信息, 用于提取样本的色彩直方图.
    target.png 是图片, 我们需要从此图中检测出跟样本色彩直方图相似的部分.
    :return:
    """
    sample = cv.imread('../dataset/data/other/sample.png')
    target = cv.imread('../dataset/data/other/target.png')
    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    tar_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    # 从样本图片中提取直方图信息. (取 roi_hsv 中的两个通道, 作直方图, 得到的就是彩色直方图).
    roihist = cv.calcHist(images=[roi_hsv], channels=[0, 1], mask=None, histSize=[180, 256], ranges=[0, 180, 0, 256])
    print(roihist.shape)
    # 规划到 0-255 之间
    cv.normalize(src=roihist, dst=roihist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    # 直方图反向投影: 计算 tar_hsv 图像中每个值在 roihist 直方图中的对应值, 并将该值对应到 dst 上.
    dst = cv.calcBackProject(images=[tar_hsv], channels=[0, 1], hist=roihist, ranges=[0, 180, 0, 256], scale=1)
    show_image(dst)
    return


def demo2():
    """彩色直方图反向投影"""
    image = cv.imread('../dataset/data/id_card/id_card_1.jpg')
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # 取 roi_hsv 中的两个通道, 作直方图, 得到的就是彩色直方图
    roihist = cv.calcHist([image], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv.normalize(roihist, roihist, 0, 255, cv.NORM_MINMAX)
    result = cv.calcBackProject([image], [0, 1], roihist, [0, 180, 0, 256], 1)
    show_image(result)
    return result


def demo3():
    """灰度直方图反向投影"""
    image = cv.imread('../dataset/data/id_card/id_card_1.jpg')
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    roihist = cv.calcHist([image], [0], None, [255], [0, 255])
    cv.normalize(roihist, roihist, 0, 255, cv.NORM_MINMAX)
    result = cv.calcBackProject([image], [0], roihist, [0, 255], 1)
    show_image(result)
    return result


if __name__ == '__main__':
    demo3()
