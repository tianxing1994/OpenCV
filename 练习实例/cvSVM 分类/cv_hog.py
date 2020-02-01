import cv2 as cv
import numpy as np


# win_size = (32, 32)
# win_stride = (16, 16)
# block_size = (16, 16)
# block_stride = (16, 16)
# cell_size = (16, 16)
# nbins = 9

win_size = (32, 32)
win_stride = (4, 4)
block_size = (16, 16)
block_stride = (4, 4)
cell_size = (4, 4)
nbins = 9

args = (win_size, block_size, block_stride, cell_size, nbins)
hog = cv.HOGDescriptor(*args)


def get_hog_dataset(target_dataset, backgrounds):
    data = []
    target = []

    roi_list, label_list, channel_list = target_dataset
    for roi, label, channel in zip(roi_list, label_list, channel_list):
        h, w = roi.shape
        h_ret = int((h - win_size[1]) // win_stride[1] + 1)
        w_ret = int((w - win_size[0]) // win_stride[0] + 1)
        descriptors = hog.compute(roi, winStride=win_stride, padding=(0, 0))
        descriptors = descriptors.reshape(h_ret * w_ret, -1)
        data.append(descriptors)
        target.append(np.array([label]))

    for background in backgrounds:
        h, w = background.shape
        h_ret = int((h - win_size[1]) // win_stride[1] + 1)
        w_ret = int((w - win_size[0]) // win_stride[0] + 1)
        n = h_ret * w_ret
        descriptors = hog.compute(background, winStride=win_stride, padding=(0, 0))
        descriptors = descriptors.reshape(n, -1)
        data.append(descriptors)
        target.append(np.zeros(shape=(n,)))

    data = np.concatenate(data).astype(np.float32)
    target = np.concatenate(target).astype(np.int)
    return data, target

