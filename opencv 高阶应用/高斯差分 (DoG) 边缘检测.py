"""
参考 <OpenCV 算法精解, 基于 Python 与 C++>

高斯拉普拉斯与高斯差分的关系
当使用 k*σ 和 σ 的方差对图像分别进行卷积后得到的两张图像相差. 得到结果图像为高斯差分.
当此 k=0.95 时, 高斯分差的值与高斯拉普拉斯的值是近似相等的.
k 作为一个参数. 用 σ 来描述高斯差分的方差, 同样地也用 σ 来描述高斯拉普拉斯的方差.
"""
import numpy as np
import cv2 as cv


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def gauss_conv(image, size, sigma):
    h, w = size
    xr, xc = np.mgrid[:1, :w]
    xc -= int((w - 1) / 2)
    # x 方向的卷积核
    xk = np.exp(-np.power(xc, 2.0) / (2 * sigma))
    xk /= np.sum(xk)
    # print(xk)

    yr, yc = np.mgrid[:h, :1]
    yr -= int((h - 1) / 2)
    # y 方向的卷积核
    yk = np.exp(-np.power(yr, 2.0) / (2 * sigma))
    yk /= np.sum(yk)
    # print(yk)

    image_xk = cv.filter2D(image, cv.CV_32F, kernel=xk, borderType=0)
    image_xk_yk = cv.filter2D(image_xk, cv.CV_32F, kernel=yk, borderType=0)
    return image_xk_yk


def dog(image, size, sigma, k=1.05):
    image_sigma = gauss_conv(image, size, sigma)
    image_sigma_k = gauss_conv(image, size, k * sigma)
    dog = image_sigma_k - image_sigma
    dog /= (np.power(sigma, 2.0) * (k-1))
    return dog


def demo1():
    image_path = '../dataset/data/image_sample/bird.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print(gray.shape)
    show_image(gray)
    # 一般来说, 卷积核取 (6σ+1, 6σ+1) 的大小, 即卷积核的大小与标准差相关, 这考虑了三倍标准差之内的值.
    result = gauss_conv(gray, size=(13, 13), sigma=2)
    result = np.array(result, dtype=np.uint8)
    print(result.shape)
    show_image(result)
    return


def demo2():
    image_path = '../dataset/data/image_sample/bird.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print(gray.shape)
    show_image(gray)

    result = dog(gray, size=(13, 13), sigma=2)
    print(result.max())
    print(result.min())
    result_image = np.array(np.where(result > 0, 255, 0), dtype=np.uint8)
    print(result.shape)
    show_image(result_image)
    return


def demo3():
    image_path = '../dataset/data/image_sample/bird.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print(gray.shape)
    show_image(gray)

    result = cv.Laplacian(gray, ddepth=cv.CV_32F, ksize=13, borderType=0)
    print(result.max())
    print(result.min())
    result_image = np.array(np.where(result > 0, 255, 0), dtype=np.uint8)
    print(result.shape)
    show_image(result_image)
    return


if __name__ == '__main__':
    # demo3()
    demo2()
