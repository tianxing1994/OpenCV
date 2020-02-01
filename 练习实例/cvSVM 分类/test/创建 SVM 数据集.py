# coding=utf8
import cv2 as cv
import numpy as np

from config import win_size, win_stride
from config import hog
from load_data.load_annotation import get_dataset


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


data = []
target = []

target_dataset, backgrounds = get_dataset()

roi_list, label_list, channel_list = target_dataset
for roi, label, channel in zip(roi_list, label_list, channel_list):
    h, w = roi.shape
    h_ret = int((h - win_size[1]) // win_stride[1] + 1)
    w_ret = int((w - win_size[0]) // win_stride[0] + 1)
    descriptors = hog.compute(roi, winStride=win_stride, padding=(0, 0))
    descriptors = descriptors.reshape(h_ret*w_ret, -1)
    data.append(descriptors)
    target.append(label)

for background in backgrounds:
    h, w = background.shape
    h_ret = int((h - win_size[1]) // win_stride[1] + 1)
    w_ret = int((w - win_size[0]) // win_stride[0] + 1)
    descriptors = hog.compute(background, winStride=win_stride, padding=(0, 0))
    descriptors = descriptors.reshape(h_ret*w_ret, -1)

    data.append(descriptors)
    target.append(2)


data = np.concatenate(data)
target = np.array(target)
print(data.shape)
print(target.shape)

