"""
相关函数:
cv2.compareHist
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def create_rgb_hist(image):
    """
    计算 rgb 图像的直方图, 将 256 分成 16 个 bins, 将 0-255 的像素值归入 0-15 的 16 个 bin 中.
    以 16 进制, RGB 有 16 * 16 * 16 种组合. 将每一种组合出现的次数计入 rgbHist 数组中.
    以此作为 rgb 图像的直方图.
    :param image:
    :return:
    """
    h, w, c = image.shape
    rgbHist = np.zeros([16*16*16, 1], np.float32)
    bin_size = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b/bin_size)*16*16 + np.int(g/bin_size)*16 + np.int(r/bin_size)
            rgbHist[np.int(index), 0] = rgbHist[np.int(index), 0] + 1
    return rgbHist


def hist_compare(image1, image2):
    """
    采用自定义的 create_rgb_hist 函数计算 rgb 图像的直方图.
    再用 cv2.compareHist 函数对比两个直方图的相似度.
    :param image1:
    :param image2:
    :return:
    """
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    # 巴氏距离: 越接近于 0 越相似
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    # 相关性: 越接近于 1 越相似
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    # 卡方: 越接近于 0 越相似
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    return match1, match2, match3


if __name__ == '__main__':

    image1 = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/rice.png")
    image2 = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/rice.png")
    # cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)

    # cv.imshow("image1", image1)
    # cv.imshow("image2", image2)
    result = hist_compare(image1, image2)
    print(result)

    # cv.waitKey(0)
    # cv.destroyAllWindows()