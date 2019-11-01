import os
import re

from template_match.config import PROJECT_PATH


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
    """
    用于判断字符串是否为空白行. 如: '\n'
    如果 string 是空字符串 '', 则将返回 False.
    :param string: 字符串, 如: '\n'
    :return:

    demo:
    string = ''
    result = is_blank(string)
    print(result)   # 返回 False
    """
    pattern = re.compile("\s")
    match = re.search(pattern, string)
    if match is not None:
        return True
    else:
        return False


def get_sample_by_label_list(cls_list, channel_list, data_path=None):
    """
    给定 label_list 获取包含这些标签的迭代器.
    :param data_path: 标注数据的 txt 文档. 内容如:
    ```
    dataset/image/snapshot_1571887242.jpg
    196, 558, 66, 69, 0
    685, 554, 78, 67, 1

    dataset/image/snapshot_1571884207.jpg
    145, 637, 93, 93, 0

    dataset/image/snapshot_1571884792.jpg
    332, 582, 71, 62, 2
    616, 583, 73, 60, 3
    ```
    每一个样本之间都会有空行隔开.
    :param cls_list: 包含示签值的列表, 如: [0, 2, 6].
    :param channel_list: 指定样来本源的渠道列表, 如: [0, 1].
    :return: [image_path, [bounding_box], [label]]

    demo:
    data_path = '../dataset/annotation.txt'
    sample = get_sample_by_label_list(label_list=[0, 2, 6])
    print(next(sample))
    print(next(sample))
    print(next(sample))
    print(next(sample))
    print(next(sample))
    print(next(sample))
    """
    if data_path is None:
        data_path = os.path.join(PROJECT_PATH, 'template_match/dataset/annotation.txt')
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
                    yield result
                result = [None, [], [], []]
                continue
            # 文档结束时, line_data 是空字符串, 三项验证都是 False, 到此处则跳出循环.
            break


dataset = get_sample_by_label_list(cls_list=[0, 2, 6], channel_list=[0, 1])


if __name__ == '__main__':
    string = '264, 547, 47, 55, 0, 1'
    result = get_mark_label(string)
    print(result)
