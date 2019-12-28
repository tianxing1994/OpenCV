"""
参考链接：
https://blog.csdn.net/wsp_1138886114/article/details/97683219
"""
import numpy as np
import cv2 as cv
from scipy.signal import convolve2d as conv2


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def flip180(src):
    ret = np.flip(src, axis=0)
    return ret


def rl_deconvblind(img, psf, iterations):
    """
    Richardson–Lucy 算法
    :param img:
    :param psf:
    :param iterations:
    :return:
    """
    img = img.astype(np.float64)
    psf = psf.astype(np.float64)
    init_img = img.copy()
    psf_hat = flip180(psf)
    for i in range(iterations):
        est_conv = conv2(init_img, psf, 'same')
        relative_blur = img / est_conv
        error_est = conv2(relative_blur, psf_hat, 'same')
        init_img = init_img * error_est
    return np.uint8(init_img)


def get_2d_gaussian_kernel(ksize, sigma):
    w, h = ksize
    p_kernel = np.multiply(cv.getGaussianKernel(h, sigma), (cv.getGaussianKernel(w, sigma)).T)
    kernel = p_kernel / np.sum(p_kernel)
    return kernel


def gaussian_blur(src, ksize, sigma):
    blurred = cv.GaussianBlur(src, ksize=ksize, sigmaX=sigma, sigmaY=sigma)
    return blurred


if __name__ == '__main__':
    path = '../../dataset/data/airport/airport1.jpg'
    image = cv.imread(path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = gaussian_blur(gray, ksize=(15, 15), sigma=5)
    show_image(blurred)
    iterations = 50
    psf = get_2d_gaussian_kernel((15, 15), 5)

    deblurred = rl_deconvblind(blurred, psf, iterations)

    show_image(deblurred)
