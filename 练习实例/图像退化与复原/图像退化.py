"""
参考链接:
https://www.cnblogs.com/lynsyklate/p/8047510.html

噪声:
高斯噪声:
高斯噪声是指它的概率密度函数服从高斯分布(即: 正态分布) 的一类噪声.
与椒盐噪声相似 (Salt And Pepper Noise), 高斯噪声 (Gausian Noise) 也是数字图像的一个常见噪声.
椒盐噪声是出现在随机位置, 噪点深度基本固定的噪声, 高斯噪声与其相反, 是几乎每个点上都出现噪声, 噪点深度随机的噪声.

"""
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def gaussian_noisy(image, sigma):
    h, w, c = image.shape
    mean = 0
    gaussian_noise = np.random.normal(loc=mean, scale=sigma, size=(h, w, c))
    noisy = image + gaussian_noise
    ret = np.array(noisy, dtype=np.uint8)
    return ret


def salt_pepper_noisy(image, s_vs_p=0.5, rate=0.004):
    out = np.copy(image)
    num_salt = np.ceil(rate * image.size * s_vs_p)
    coords = tuple([np.random.randint(0, i - 1, int(num_salt)) for i in image.shape])
    out[coords] = 255
    num_pepper = np.ceil(rate * image.size * (1. - s_vs_p))
    coords = tuple([np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape])
    out[coords] = 0
    return out


def demo1():
    # sigma = 25 的高斯噪声图
    image_path = "../../dataset/data/airport/airport1.jpg"
    image = cv.imread(image_path)
    show_image(image)

    g_image = gaussian_noisy(image, sigma=25)
    show_image(g_image)
    return


def demo2():
    # 椒盐噪声
    image_path = "../../dataset/data/airport/airport1.jpg"
    image = cv.imread(image_path)
    show_image(image)

    g_image = salt_pepper_noisy(image)
    show_image(g_image)
    return


if __name__ == '__main__':
    # demo1()
    demo2()






















