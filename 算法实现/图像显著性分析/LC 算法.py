"""
参考链接:
https://asdfv1929.github.io/2018/05/11/saliency-LC/

LC 图像显著性分析.
计算某个像素在整个图像上的全局对比度, 即该像素与图像中其他所有像素在颜色上的距离之和作为该像素的显著值.


"""
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def calc_dist(hist):
    """
    计算每一个灰度值与图像中所有其它像素值的距离之和.
    :param hist:
    :return:
    """
    dist = {}
    for gray_value in range(256):
        value = 0.0
        for k in range(256):
            value += hist[k][0] * abs(gray_value - k)
        dist[gray_value] = value
    return dist


def lc_significant_image(image, dist_dict):
    """
    :param image: 灰度图像
    :param dist_dict:
    :return:
    """
    h, w = image.shape
    dst = np.zeros(shape=(h, w))
    for i in range(h):
        for j in range(w):
            temp = image[i, j]
            dst[i, j] = dist_dict[temp]
    result = (dst - np.min(dst)) / (np.max(dst) - np.min(dst))
    result = np.array(result * 255, dtype=np.uint8)
    return result


def demo1():
    image_path = '../../dataset/data/image_sample/bird.jpg'

    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    hist_array = cv.calcHist([gray], [0], None, [256], [0.0, 256.0])
    dist_dict = calc_dist(hist_array)
    result = lc_significant_image(image=gray, dist_dict=dist_dict)
    _, binary = cv.threshold(result, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    show_image(binary)
    return


def demo2():
    """LC 算法逻辑的原意代码. 我感觉是这样啊的, 不知道为什么结果不一样. """
    image_path = '../../dataset/data/image_sample/bird.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    result = np.zeros(shape=(h, w))
    for i in range(h):
        for j in range(w):
            print(i, j)
            result[i, j] = np.sum(np.abs(gray[i, j] - gray))

    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    show_image(result)
    return


if __name__ == '__main__':
    demo1()
    # demo2()
