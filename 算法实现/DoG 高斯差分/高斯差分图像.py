import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def getGaussianKernel(sigma, size):
    h, w = size
    r, c = np.mgrid[0:h:1, 0:w:1]
    r -= int((h - 1) / 2)
    c -= int((w - 1) / 2)
    sigma_ = pow(sigma, 2.0)
    norm_ = np.power(r, 2.0) + np.power(c, 2.0)
    gaussianKernel = np.exp(- norm_ / (2 * sigma_))
    gaussianKernel /= np.sum(gaussianKernel)
    return gaussianKernel


def demo1():
    """LoG 高斯拉普拉斯"""
    image_path = "../../dataset/data/image_sample/lena.png"
    image = cv.imread(image_path)
    ksize = 9
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    kernel1 = getGaussianKernel(sigma, (ksize, ksize))
    image1 = cv.filter2D(image, cv.CV_64F, kernel=kernel1)
    image1 = np.array(image1, dtype=np.uint8)
    # show_image(image1)
    kernel2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # kernel2 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    image2 = cv.filter2D(image1, cv.CV_64F, kernel2)
    image2 = np.abs(image2)
    print(np.max(image2))
    image2 = np.array(image2 / np.max(image2) * 255, dtype=np.uint8)
    show_image(image2)
    return


def demo2():
    """DoG 高斯差分"""
    image_path = "../../dataset/data/image_sample/lena.png"
    image = cv.imread(image_path)
    ksize = 9
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    kernel1 = getGaussianKernel(1.05 * sigma, (ksize, ksize))
    kernel2 = getGaussianKernel(sigma, (ksize, ksize))
    # kernel1 = getGaussianKernel(sigma, (ksize, ksize))
    # kernel2 = getGaussianKernel(0.95 * sigma, (ksize, ksize))
    image1 = cv.filter2D(image, cv.CV_64F, kernel=kernel1)
    image2 = cv.filter2D(image, cv.CV_64F, kernel=kernel2)
    image = np.abs(image2 - image1)
    print(np.max(image))
    image = np.array(image / np.max(image) * 255, dtype=np.uint8)
    show_image(image)
    return


if __name__ == '__main__':
    demo1()
    demo2()
