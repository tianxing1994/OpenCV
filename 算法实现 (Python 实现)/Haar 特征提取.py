"""
参考链接:
https://blog.csdn.net/zouxy09/article/details/7929570
https://blog.csdn.net/u011913612/article/details/78491896

现在这个程序, 运行时 MemoryError. 这个逻辑不切实际呀.
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def integral(image):
    """计算积分图"""
    result = np.zeros(shape=image.shape, dtype=np.int32)
    for r in range(image.shape[0]):
        accumulate = 0
        for c in range(image.shape[1]):
            accumulate += image[r, c]
            result[r, c] = result[r-1, c] + accumulate
    return result


# Haar 提特征提取框:
# ---------
# |   |   |
# | - | + |
# |   |   |
# ---------


def get_haar_a_features_area(height, width):
    """
    以如上的 Haar-like 特征提取框为例. 对于图片形状为 (height, width),
    则特征提取框的高度和宽度取值范围为: height_limit∈[1, height-1], width_limit∈[1, width//2-1].
    即: 穷举此范围内的所有特征提取框大小.
    在特征框大小 (h, 2*w) 确定的情况下, 特征框左上角的坐标 (x, y) 可能的取值为: x∈[0, height-h], y∈[0, width-2*w]
    注: 这里把特征框的一半作为宽度.
    :param height:
    :param width:
    :return:
    """
    height_limit = height
    width_limit = width // 2
    features = list()
    for h in range(1, height_limit):
        for w in range(1, width_limit):
            h_move_limit = height - h
            w_move_limit = width - 2*w
            for x in range(h_move_limit):
                for y in range(w_move_limit):
                    features.append((x, y, h, w))
    return features


def calc_haar_a_features(integral_graph, features_graph):
    haar_features = list()
    for x, y, h, w in features_graph:
        haar1 = integral_graph[x, y] + integral_graph[x+h, y+w] - integral_graph[x, y+w] - integral_graph[x+h, y]
        haar2 = integral_graph[x, y+w] + integral_graph[x+h, y+2*w] - integral_graph[x, y+2*w] - integral_graph[x+h, y+w]
        haar_features.append(haar2 - haar1)
    return haar_features


if __name__ == '__main__':
    image_path = r"C:\Users\Administrator\PycharmProjects\OpenCV\dataset\image0.JPG"
    image = cv.imread(image_path)
    integral_graph = integral(image)
    features_graph = get_haar_a_features_area(image.shape[0], image.shape[1])
    result = calc_haar_a_features(integral_graph, features_graph)
    print(result)
