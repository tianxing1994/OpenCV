"""
参考链接:
https://blog.csdn.net/qq_41686130/article/details/81229353
"""
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def process(img):
    # 高斯平滑
    gaussian = cv.GaussianBlur(img, (3, 3), 0, 0, cv.BORDER_DEFAULT)
    # 中值滤波
    median = cv.medianBlur(gaussian, 5)
    # Sobel算子
    # 梯度方向: x
    sobel = cv.Sobel(median, cv.CV_8U, 1, 0, ksize=3)
    # 二值化
    ret, binary = cv.threshold(sobel, 170, 255, cv.THRESH_BINARY)
    # 核函数
    element1 = cv.getStructuringElement(cv.MORPH_RECT, (9, 1))
    element2 = cv.getStructuringElement(cv.MORPH_RECT, (9, 7))
    # 膨胀
    dilation = cv.dilate(binary, element2, iterations=1)
    # 腐蚀
    erosion = cv.erode(dilation, element1, iterations=1)
    # 膨胀
    dilation2 = cv.dilate(erosion, element2, iterations=3)
    return dilation2


def get_region(img):
    regions = []
    # 查找轮廓
    _, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv.contourArea(contour)
        if (area < 2000):
            continue
        eps = 1e-3 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, eps, True)
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        ratio = float(width) / float(height)
        if (ratio < 5 and ratio > 1.8):
            regions.append(box)
    return regions


def detect(image):
    # 灰度化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    prc = process(gray)
    regions = get_region(prc)
    for box in regions:
        cv.drawContours(image, [box], 0, (0, 255, 0), 2)
    return image


if __name__ == '__main__':
    # 输入的参数为图片的路径
    # image_path = '../dataset/data/car_license_plate/license_plate_1.jpg'
    # image_path = '../dataset/data/car_license_plate/license_plate_2.jpg'
    image_path = '../dataset/data/car_license_plate/license_plate_3.jpg'
    image = cv.imread(image_path)
    image = detect(image)
    show_image(image)
