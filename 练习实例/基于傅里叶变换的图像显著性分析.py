"""
显著性检测的方法有很多, 此处介绍, 基于幅度谱残差的方法.
第一步: 计算图像的快速傅里叶变换矩阵 F.
第二步: 计算傅里叶变换的幅度谱的灰度级 gray_spectrum.
第三步: 计算相位谱 phase_spectrum, 然后根据相位谱计算对应的正弦谱和余弦谱.
第四步: 对第二步计算出的灰度级进行均值平滑, 记为 f_mean(gray_spectrum).
第五步: 计算谱残差 (spectral_residual). 谱残差的定义是第二步得到的幅度谱的灰度级减去第四步得到的均值平滑结果,
即: spectral_residual = gray_spectrum - f_mean(gray_spectrum)
第六步: 对谱残差进行幂指数运算 exp(spectral_residual), 即对谱残差矩阵中的每一个值进行指数运算.
第七步: 将第六步得到的幂指数作为新的 "幅度谱", 仍然使用原图的相位谱, 根据新的 "幅度谱" 和相位谱进行傅里叶变换, 可得到一个复数矩阵.
第八步: 对于第七步得到的复数矩阵, 计算该矩阵的实部和虚部的平方和的开方, 然后进行高斯平滑, 最后进行灰度级的转换 即得到显著性.

问题:
还需要一种方法, 把显著性的区域分割出来.
"""
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo():
    # 1. 灰度化读取文件，
    # image_path = '../dataset/data/exercise_image/image_text_r.png'
    # image_path = '../dataset/data/exercise_image/express_paper_1.jpg'
    image_path = '../dataset/data/exercise_image/express_paper_2.jpg'
    # image_path = '../dataset/data/exercise_image/express_paper_3.jpg'

    image = cv.imread(image_path, 0)
    show_image(image)
    h, w = image.shape

    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)

    # 幅度谱, 相位谱
    gray_spectrum = np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
    phase_spectrum = np.arctan(dft[:, :, 1], dft[:, :, 0])

    gray_spectrum_mean = cv.blur(gray_spectrum, ksize=(3, 3))
    spectral_residual = np.exp(gray_spectrum - gray_spectrum_mean)

    # 余弦谱, 正弦谱
    cos_spectrum = np.expand_dims(np.cos(phase_spectrum) * spectral_residual, axis=2)
    sin_spectrum = np.expand_dims(np.sin(phase_spectrum) * spectral_residual, axis=2)

    new_dft = np.concatenate([cos_spectrum, sin_spectrum], axis=2)
    idft = cv.dft(new_dft, cv.DFT_INVERSE)
    idft_magnitude = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
    idft_magnitude = np.array(idft_magnitude / idft_magnitude.max() * 255, dtype=np.uint8)

    show_image(idft_magnitude)
    return


def significant_image(image):
    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)

    # 幅度谱, 相位谱
    gray_spectrum = np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
    phase_spectrum = np.arctan(dft[:, :, 1], dft[:, :, 0])

    gray_spectrum_mean = cv.blur(gray_spectrum, ksize=(3, 3))
    spectral_residual = np.exp(gray_spectrum - gray_spectrum_mean)

    # 余弦谱, 正弦谱
    cos_spectrum = np.expand_dims(np.cos(phase_spectrum) * spectral_residual, axis=2)
    sin_spectrum = np.expand_dims(np.sin(phase_spectrum) * spectral_residual, axis=2)

    new_dft = np.concatenate([cos_spectrum, sin_spectrum], axis=2)
    idft = cv.dft(new_dft, cv.DFT_INVERSE)
    idft_magnitude = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
    idft_magnitude = np.array(idft_magnitude / idft_magnitude.max() * 255, dtype=np.uint8)

    return idft_magnitude


def significant_image_demo():
    # 1. 灰度化读取文件，
    image_path = '../dataset/data/exercise_image/image_text_r.png'
    # image_path = '../dataset/data/exercise_image/express_paper_1.jpg'
    # image_path = '../dataset/data/exercise_image/express_paper_2.jpg'
    # image_path = '../dataset/data/exercise_image/express_paper_3.jpg'

    image = cv.imread(image_path, 0)
    idft_magnitude = significant_image(image)
    show_image(idft_magnitude)

    idft_magnitude_blur = cv.blur(idft_magnitude, ksize=(15, 15))
    _, binary = cv.threshold(idft_magnitude_blur, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # binary_dilate = cv.dilate(binary, kernel)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # binary_erode = cv.erode(binary_dilate, kernel)
    show_image(binary)
    return


if __name__ == '__main__':
    significant_image_demo()
