"""
https://github.com/raochin2dev/RegionMerging
https://github.com/manasiye/Region-Merging-Segmentation/blob/master/Q2a.py
https://github.com/pranavgadekar/recursive-region-merging
https://github.com/CQ-zhang-2016/some-algorithms-about-digital-image-process
https://github.com/dtg67/SliceAndLabel/blob/master/SliceandLabel.py
https://blog.csdn.net/qq_19531479/article/details/79649227
"""
import cv2 as cv
import numpy as np


def show_image(image):
    cv.namedWindow('input image', cv.WINDOW_NORMAL)
    cv.imshow('input image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


image_path = '../dataset/data/fruits.jpg'
image = cv.imread(image_path)

show_image(image)

