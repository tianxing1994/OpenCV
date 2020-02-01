# coding=utf8
import os
import re
import cv2 as cv
import numpy as np

from load_data.roi_jitter import panning_enhance, gray_jitter


def show_image(image, win_name='input image'):
    # cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def is_path(string):
    """
    用于判断字符串是否是图片的相对路径.
    :param string: 字符串, 如: 'dataset/image/snapshot_1571887242.jpg'
    :return:
    """
    pattern = re.compile(".*\.jpg")
    match = re.search(pattern, string)
    if match is not None:
        return True
    else:
        return False


def is_mark(string):
    """
    :param string: 字符串, 如: '264, 547, 47, 55, 0'
    :return:
    """
    pattern = re.compile("\d+[, \d]+\d+")
    match = re.search(pattern, string)
    if match is not None:
        return True
    else:
        return False


def get_mark_bounding_box(string):
    """
    如果一个字符串通过 is_mark 返回为 True, 即已确定为一个标记行.
    则用此函数从中获取其标记的 bounding box 的值.
    :param string: 字符串, 如: '264, 547, 47, 55, 0, 1'
    :return: bounding box. (x, y, w, h)
    demo:
    string = '264, 547, 47, 55, 0, 1'
    result = get_mark_bounding_box(string)
    print(result)
    """
    pattern = re.compile("(\d+, \d+, \d+, \d+), [, \d]+")
    match = re.search(pattern, string)
    if match is None:
        return None
    position_string = match.group(1)
    result = position_string.split(', ')
    result = list(map(lambda x: int(x), result))
    return result


def get_mark_label(string):
    """
    如果一个字符串通过 is_mark 返回为 True, 即已确定为一个标记行.
    则用此函数从中获取其标记的标签值.
    :param string: 字符串, 如: '264, 547, 47, 55, 0, 1'
    :return:
    demo:
    string = '264, 547, 47, 55, 0, 1'
    result = get_mark_label(string)
    print(result)
    """
    pattern = re.compile("(?:\d+, ){4}(\d+)[, \d]*")
    match = re.search(pattern, string)
    if match is None:
        return None
    result = int(match.group(1))
    return result


def get_channel_label(string):
    """
    如果一个字符串通过 is_mark 返回为 True, 即已确定为一个标记行.
    则用此函数从中获取其渠道标签值.
    :param string: 字符串, 如: '264, 547, 47, 55, 0, 1'
    :return:
    demo:
    string = '264, 547, 47, 55, 0, 1'
    result = get_channel_label(string)
    print(result)
    """
    pattern = re.compile("(?:\d+, ){5}(\d+)")
    match = re.search(pattern, string)
    if match is None:
        return None
    result = int(match.group(1))
    return result


def is_blank(string):
    if len(string) == 0:
        return True
    else:
        return False


def get_sample_by_label_list(cls_list, channel_list, data_path=None):
    """
    给定 label_list 获取包含这些标签的迭代器.
    :param data_path: 标注数据的 txt 文档. 内容如:
    ```
    dataset/image/luosi.jpg
    79, 256, 35, 31, 0, 0
    209, 324, 37, 29, 0, 0
    337, 250, 34, 36, 1, 0
    470, 321, 37, 33, 1, 0
    599, 249, 36, 34, 0, 0
    738, 317, 33, 33, 0, 0
    ```
    每一个样本之间都会有空行隔开.
    :param cls_list: 包含示签值的列表, 如: [0, 2, 6].
    :param channel_list: 指定样来本源的渠道列表, 如: [0, 1].
    :return: [image_path, [bounding_box], [label]]
    demo:
    dataset = get_sample_by_label_list(cls_list=[0], channel_list=[0])
    for data in dataset:
        print(type(data))
        print(data)
    """
    if data_path is None:
        # p = os.path.dirname(__file__)
        # data_path = os.path.join(p, "dataset/annotation.txt")
        data_path = "dataset/annotation.txt"
    result = [None, [], [], []]
    with open(data_path, 'r', encoding='utf-8') as f:
        while True:
            line_data = f.readline()
            if is_path(line_data):
                result[0] = line_data.strip()
                continue
            if is_mark(line_data):
                bounding_box = get_mark_bounding_box(line_data)
                cls_label = get_mark_label(line_data)
                channel_label = get_channel_label(line_data)
                if cls_label in cls_list and channel_label in channel_list:
                    result[1].append(bounding_box)
                    result[2].append(cls_label)
                    result[3].append(channel_label)
                continue
            if is_blank(line_data):
                if len(result[1]) != 0:
                    result[1] = np.array(result[1])
                    result[2] = np.array(result[2])
                    result[3] = np.array(result[3])
                    yield result
                    result = [None, [], [], []]
                    continue
            # 文档结束时, line_data 是空字符串, 三项验证都是 False, 到此处则跳出循环.
            break
