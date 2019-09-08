"""
参考链接:
https://blog.csdn.net/u010006643/article/details/46417091
http://blog.csdn.net/sad_123_happy/article/details/9087989

YALE人脸数据库（美国，耶鲁大学） 数据下载不成功, 不能按参考链接中进行实现.
http://cvc.yale.edu/projects/yalefacesB/yalefacesB.html
https://blog.csdn.net/duan19920101/article/details/50679061

LBP 算法

圆形 LBP 算子
如:
[[1, 2, 2],
 [9, 5, 6],
 [5, 3, 1]]

[[0, 0, 0],
 [1,  , 1],
 [1, 0, 0]]

 binary = 00010011
 decimal = 19

该算子将像素八邻域的像素值大于中心点值的标注为 1, 反之标注为 0. 这样我们就会得到一个八位数. 这时从左上角开始算最高位,
最后左边的为最低位, 在这个图中就是 00010011, 换算十进制就是 19, 我们就把这个值作为中心点的 LBP 算子值.

为了让 LBP 特值取有旋转不变性, 将二进制串进行旋转. 假设一开始得到的 LBP 特征为 10010000,
那么将这个二进制特征按照顺时针方向旋转, 可以转化为 00001001 的形式, 这样得到的 LBP 值是最小的.
无论图像怎么旋转, 对点提取的二进制特征的最小值是不变的. 用最小值作为提取的 LBP 特征, 这样 LBP 就是旋转不变的.

其他:
之前用过 opencv 的模板匹配, 其原理是两种图片对应像素相减, 差值的平方求和, 作为差异度.
这种方法要求两张图片完全一样, 只要两图的结构稍有不同就不准确了. 只适合找出相同的图片.
而现在用 lbp 特征提取, 我有一个体悟, 只要我们用这种基于当前像素及其周围像素的统计特征. 即可改善这种缺点.
因为只要需要匹配的重要特征正好在特征提取窗口内即可, 而不需要两图像素完全对齐.
未经验证. 稍有感悟.
"""
import os
import math
import numpy as np
import cv2 as cv


def calc_lbp_hist(lbp_image):
    """
    把 LBP 图像分为 7*4 份, 分别计算其统计直方图. 横轴数据类型为 0-255 之间, 值为各值出现的频率.
    :param lbp_image: ndarray, 其值为 lbp 特征. 目前我们要使用的数据集图像大小为 (116, 98).
    :return: ndarray, 形状为 (4*7, 256)
    """
    result = np.zeros(shape=(4*7, 256))
    h, w = lbp_image.shape
    h_i = h / 7
    w_i = w / 4
    row = 0
    for i in range(7):
        for j in range(4):
            roi = lbp_image[int(i*h_i):int((i+1)*h_i),
                            int(j*w_i):int((j+1)*w_i)]
            hist = cv.calcHist([np.array(roi, np.uint8)], [0], None, [256], [0, 256])
            result[row] = hist.T
            row += 1
    return result


def calc_similarity(lbp_hist1, lbp_hist2):
    """
    如 calc_lbp_hist 计算, 我们将 lbp_image 分成 7*4 份, 结果 lbp_hist 为 7*4 行 256 列.
    两 lbp_hist 相减再平方求和, 得到的值作为两图的差异度.
    """
    return np.sum((lbp_hist1 - lbp_hist2)**2)


def local_binary_pattern(image, r=2):
    """
    提取图像的 LBP 特征.
    此处取图像像素的 8 邻域 LBP 特征. 得到的将是 8 位 0/1 值, 转换为十进制则为 0-255之间.
    :param image: 输入灰度图像
    :param r: 指定 LBP 特征提取窗口半径大小.
    :return: 返回与 image 形状大小一样的图像, 取值 0-255 之间.
    """
    # LBP 算子 8 邻域像素相对位置索引. 左上角为起点.
    region8_i = [-1, 0, 1, 1, 1, 0, -1, -1]
    region8_j = [-1, -1, -1, 0, 1, 1, 1, 0]
    result = np.zeros(shape=image.shape)

    h, w = image.shape
    for i in range(r, h-r):
        for j in range(r, w-r):
            binary_lbp = ''
            center_pixel = image[i, j]
            for relative_i, relative_j in zip(region8_i, region8_j):
                i_ = i + relative_i
                j_ = j + relative_j
                if image[i_, j_] > center_pixel:
                    binary_lbp += '1'
                else:
                    binary_lbp += '0'
            binary_lbp_striped = binary_lbp.strip('0') or '0'
            result[i, j] = int(binary_lbp_striped, base=2)
    return result


if __name__ == '__main__':
    image_path = r"C:\Users\Administrator\PycharmProjects\OpenCV\dataset\image0.JPG"
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    result = local_binary_pattern(gray)
    print(result.shape)
    print(result)
    print(result.max())



















