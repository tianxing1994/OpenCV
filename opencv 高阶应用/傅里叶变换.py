"""
参考链接:
https://v.youku.com/v_show/id_XMzE0NDUyNDE0OA
http://www.doc88.com/p-9062564186170.html
https://i.youku.com/xuxiaozhan

"""
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def fourier_demo1():
    """傅里叶变换, 并显示幅度谱, 不知道它为什么是反的."""
    image = cv.imread('../dataset/data/exercise_image/image_text_r.png', 0)

    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)

    dft_magnitude = 20 * np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
    dft_magnitude = np.array(dft_magnitude / dft_magnitude.max() * 255, dtype=np.uint8)
    show_image(dft_magnitude)

    # 逆变换
    idft = cv.dft(dft, cv.DFT_INVERSE)
    idft_magnitude = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
    idft_magnitude = np.array(idft_magnitude / idft_magnitude.max() * 255, dtype=np.uint8)
    show_image(idft_magnitude)
    return


def fourier_demo2():
    """
    傅里叶变换中心化, 并显示幅度谱. 我不知道为什么中间会有一条竖线.
    幅度谱中心化原理
    参考链接:
    https://blog.csdn.net/qq_36607894/article/details/92809731
    """
    image = cv.imread('../dataset/data/exercise_image/image_text_r.png', 0)

    # 傅里叶变换, 计算幅度谱并显示
    h, w = image.shape
    center_array = np.array([np.power(-1, r + c) for r in range(h) for c in range(w)]).reshape(image.shape)
    center_image = image * center_array
    dft = cv.dft(np.float32(center_image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_magnitude = np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
    dft_magnitude = np.array(dft_magnitude / dft_magnitude.max() * 255, dtype=np.uint8)
    show_image(dft_magnitude)
    return


def fourier_demo3():
    """傅里叶变换中心化, 后的逆变换. 不知道它为什么是反的."""
    image = cv.imread('../dataset/data/exercise_image/image_text_r.png', 0)

    # 傅里叶变换中心化
    h, w = image.shape
    center_array = np.array([np.power(-1, r + c) for r in range(h) for c in range(w)]).reshape(image.shape)
    center_image = image * center_array
    dft = cv.dft(np.float32(center_image), flags=cv.DFT_COMPLEX_OUTPUT)

    # 显示幅度谱
    dft_shift_magnitude = 20 * np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
    dft_shift_magnitude = np.array(dft_shift_magnitude / dft_shift_magnitude.max() * 255, dtype=np.uint8)
    show_image(dft_shift_magnitude)

    # 逆变换
    idft = cv.dft(dft, cv.DFT_INVERSE)
    idft_magnitude = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
    idft_magnitude = np.array(idft_magnitude / idft_magnitude.max() * 255, dtype=np.uint8)

    show_image(idft_magnitude)
    return


def fourier_demo4():
    """傅里叶变换中心化, 后的逆变换. 不知道它为什么是反的."""
    image = cv.imread('../dataset/data/exercise_image/image_text_r.png', 0)

    # 傅里叶变换中心化
    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)

    # 显示幅度谱
    dft_shift_magnitude = 20 * np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
    dft_shift_magnitude = np.array(dft_shift_magnitude / dft_shift_magnitude.max() * 255, dtype=np.uint8)
    show_image(dft_shift_magnitude)

    # 逆变换
    idft = cv.dft(dft, cv.DFT_INVERSE)
    idft_magnitude = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
    idft_magnitude = np.array(idft_magnitude / idft_magnitude.max() * 255, dtype=np.uint8)

    show_image(idft_magnitude)
    return


if __name__ == '__main__':
    fourier_demo4()
