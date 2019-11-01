"""
这个需要逐像素遍历, 很慢.

参考链接:
jianshu.com/p/1be50bd846ed
"""
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def square_distance(vector1, vector2):
    return np.sum(np.square(vector1 - vector2))


def get_salient(image, position, scale):
    h, w, c = image.shape
    i, j = position
    p = image[i, j]
    i1 = np.clip(i - scale, 0, h)
    i2 = np.clip(i + scale, 0, h)
    j1 = np.clip(j - scale, 0, h)
    j2 = np.clip(j + scale, 0, h)
    roi = image[i1: i2, j1: j2]
    m = np.mean(roi, axis=(0, 1))
    s = square_distance(p, m)
    return s


def ac_salient(image):
    h, w, c = image.shape
    scale = np.min([h, w])
    scale_8 = int(scale / 8)
    scale_4 = int(scale / 4)
    scale_2 = int(scale / 2)

    salient_map = np.zeros(shape=(h, w))
    for i in range(h):
        for j in range(w):
            s_8 = get_salient(image, (i, j), scale_8)
            s_4 = get_salient(image, (i, j), scale_4)
            s_2 = get_salient(image, (i, j), scale_2)
            s = s_8 + s_4 + s_2
            salient_map[i, j] = s

    smin = np.min(salient_map)
    smax = np.max(salient_map)
    result = (salient_map - smin) / (smax - smin)
    result = np.array(255 * result, dtype=np.uint8)
    return result


if __name__ == '__main__':
    image_path = '../../dataset/data/image_sample/bird.jpg'
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    result = ac_salient(image)
    show_image(result)
