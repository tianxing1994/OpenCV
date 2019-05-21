"""
相关函数:
cv2.blur
cv2.medianBlur
cv2.GaussianBlur
cv2.filter2D
"""
import cv2 as cv
import numpy as np


def blur_demo(image):
    """
    均值模糊: 用大小为 ksize 的卷积核去扫描图片(如 ksize=(15,1), 则用 (15,1) 的卷积核),
    取核上的均值并向下取整后作为当前像素的卷积后的值.
    :param image:
    :return:
    """
    result = cv.blur(image, ksize=(15,1))
    return result


def median_blur_demo(image):
    """
    中值模糊: 用大小为 ksize 的方向卷积核去扫描图片(如 ksize=3, 则用 3*3 的卷积核),
    对核上的值进行排序, 后取中间位置的值作为当前像素的值.
    :param image:
    :return:
    """
    result = cv.medianBlur(image, ksize=5)
    return result


def gaussian_blur_demo(image):
    """
    高斯模糊:
    :param image:
    :return:
    """
    result = cv.GaussianBlur(image, (0, 0), 5)
    return result


def custom_blur_demo(image):
    """
    自定义中值滤波: 自定义一个 kernel 卷积核(形状为正方形, 单边为奇数, 所有值的和为 1),
    对图像进行 filter2D 进行卷积操作.
    :param image:
    :return:
    """
    # 5*5 的均值卷积核
    # kernel = np.ones([5, 5], np.float32) / 25
    # 锐化卷积核
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    result = cv.filter2D(image, -1, kernel=kernel)
    return result


def clamp(pixel):
    if pixel>255:
        return 255
    elif pixel<0:
        return 0
    else:
        return pixel


def gaussian_noise(image):
    """
    添加噪声: 遍历图片中的每一个像素, 为其添加高斯噪声.
    :param image:
    :return:
    """
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]  # blue
            g = image[row, col, 1]  # green
            r = image[row, col, 2]  # red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    return image


if __name__ == '__main__':
    image_path = 'C:/Users/tianx/PycharmProjects/opencv/dataset/lena.png'
    src = cv.imread(image_path)
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)

    src = gaussian_blur_demo(src)

    cv.imshow("input image1", src)
    cv.waitKey(0)
    cv.destroyAllWindows()