import os
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image', flags=cv.WINDOW_NORMAL):
    cv.namedWindow(win_name, flags)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def resize_image(image, to_size):
    """
    将图像 resize 到固定大小, 并返回在宽度与高度上的比例
    需要适应原图的长宽比例, 原图长边 resize 后依然是长边.
    :param image:
    :param to_size:
    :return:
    """
    h, w = image.shape[:2]
    to_w, to_h = to_size
    h_radio = to_h / h
    w_radio = to_w / w
    image_resized = cv.resize(image, dsize=(0, 0), fx=w_radio, fy=h_radio)
    return image_resized, h_radio, w_radio


def calc_canny_edge(image):
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    edge = cv.Canny(gray, 50, 150)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    edge = cv.dilate(edge, kernel)
    return edge
