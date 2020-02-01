# coding=utf8
import os
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    # cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def get_roi_by_bounding_box(gray, bounding_box, angle):
    """
    将灰度图像先旋转再截取 ROI.
    :param gray:
    :param bounding_box:
    :param angle: 指将图像逆时针旋转的角度.
    :return:
    """
    x, y, w, h = bounding_box
    center_x = int(x + w // 2)
    center_y = int(y + h // 2)
    center = (center_x, center_y)

    h_, w_ = gray.shape
    m = cv.getRotationMatrix2D(center=center, angle=angle, scale=1)
    gray_rotated = cv.warpAffine(gray, M=m, dsize=(w_, h_))
    result = gray_rotated[y:y + h, x:x + w]
    return result


def panning_enhance(data):
    """
    对原 ROI 平移 8 邻域内平移以产生新的数据.
    :param data: 需要平移增强的数据. 如:
    ['dataset/image/luosi.jpg',
    [[79, 256, 35, 31],
     [209, 324, 37, 29],
     [599, 249, 36, 34] ,
     [738, 317, 33, 33]],
    [0, 0, 0, 0],
    [0, 0, 0, 0]]
    :return:
    """
    result = [None, [], [], []]
    path, roi_list, label_list, channel_list = data
    result[0] = path
    for roi, label, channel in zip(roi_list, label_list, channel_list):
        # for i in range(-1, 2):
        #     for j in range(-1, 2):
        for i in range(-3, 4):
            for j in range(-3, 4):
                new_roi = roi + np.array([i, j, 0, 0])
                result[1].append(new_roi)
                result[2].append(label)
                result[3].append(channel)
    result[1] = np.array(result[1])
    result[2] = np.array(result[2])
    result[3] = np.array(result[3])
    return result


def gray_jitter(data):
    """
    对图像进行旋转增强.
    :param data:
    :return:
    """
    result = [[], [], []]
    path, roi_list, label_list, channel_list = data
    # p = os.path.dirname(__file__)
    # image_path = os.path.join(p, path)
    image_path = path
    gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    for roi, label, channel in zip(roi_list, label_list, channel_list):
        for i in range(-10, 11):
            new_roi = get_roi_by_bounding_box(gray=gray, bounding_box=roi, angle=i)
            # show_image(new_roi)
            result[0].append(new_roi)
            result[1].append(label)
            result[2].append(channel)

    result[0] = np.array(result[0])
    result[1] = np.array(result[1])
    result[2] = np.array(result[2])
    return result


if __name__ == '__main__':
    from load_data.load_annotation import get_sample_by_label_list

    dataset = get_sample_by_label_list(cls_list=[0], channel_list=[0])
    for data in dataset:
        data = panning_enhance(data)
        data = gray_jitter(data)
        # print(data[0])
        # print(data[1])
        # print(data[2])
        print(len(data[0]))
        for d in data[0]:
            print(d.shape)
