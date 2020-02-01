# coding=utf8
import os

import cv2 as cv
import numpy as np

from load_data.load_annotation import get_sample_by_label_list
from load_data.roi_jitter import panning_enhance, gray_jitter


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def get_image_dataset(cls_list=(1, 2), channel_list=(0,)):
    target_dataset = [[], [], []]
    backgrounds = []

    dataset = get_sample_by_label_list(cls_list=cls_list, channel_list=channel_list)

    for data in dataset:
        path, roi_list, label_list, channel_list = data
        # p = os.path.dirname(__file__)
        # image_path = os.path.join(p, path)
        image_path = path
        gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        for roi in roi_list:
            x, y, w, h = roi
            gray[y: y+h, x: x+w] = 0
        backgrounds.append(gray)
        # show_image(gray)
        data = panning_enhance(data)
        data = gray_jitter(data)
        target_dataset[0].append(data[0])
        target_dataset[1].append(data[1])
        target_dataset[2].append(data[2])
    target_dataset[0] = np.concatenate(target_dataset[0], axis=0)
    target_dataset[1] = np.concatenate(target_dataset[1], axis=0)
    target_dataset[2] = np.concatenate(target_dataset[2], axis=0)
    return target_dataset, backgrounds
