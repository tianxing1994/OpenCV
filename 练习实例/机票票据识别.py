"""
去除票据上的波纹线. 失败.
"""
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def significant_image(image):
    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)

    # 幅度谱, 相位谱
    gray_spectrum = np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
    phase_spectrum = np.arctan2(dft[:, :, 1], dft[:, :, 0])

    gray_spectrum_mean = cv.blur(gray_spectrum, ksize=(3, 3))
    spectral_residual = np.exp(gray_spectrum - gray_spectrum_mean)

    # 余弦谱, 正弦谱
    cos_spectrum = np.expand_dims(np.cos(phase_spectrum) * spectral_residual, axis=2)
    sin_spectrum = np.expand_dims(np.sin(phase_spectrum) * spectral_residual, axis=2)

    new_dft = np.concatenate([cos_spectrum, sin_spectrum], axis=2)
    idft = cv.dft(new_dft, cv.DFT_INVERSE)
    gray_spectrum_result = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
    gray_spectrum_result = np.array(gray_spectrum_result / gray_spectrum_result.max() * 255, dtype=np.uint8)
    return gray_spectrum_result


def get_paper_corner(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    significant_gray = significant_image(gray)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    significant_gray_tohat = cv.morphologyEx(significant_gray, cv.MORPH_TOPHAT, kernel)

    _, binary = cv.threshold(significant_gray_tohat, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 15))
    dst = cv.dilate(binary, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
    dst = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 15))
    dst = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel)

    _, contours, _ = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    max_contour_area = 0
    max_contour = None
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > max_contour_area:
            max_contour = contour
            max_contour_area = area

    # 估计出四边形.
    approx = cv.approxPolyDP(curve=max_contour,
                             epsilon=200,
                             closed=True)
    return approx[:, 0]


def perspective_transformation(image, src_nd):
    h, w = image.shape[:2]

    src_nd = np.array(src_nd, dtype=np.float32)
    dst_nd = np.array([[h, 0], [0, 0], [0, w], [h, w]], dtype=np.float32)
    perspective_matrix = cv.getPerspectiveTransform(src=src_nd,
                                                    dst=dst_nd)
    result = cv.warpPerspective(src=image, M=perspective_matrix, dsize=(h, w), flags=cv.INTER_NEAREST)
    return result


def dft_transformation(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dft = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)
    return dft


def get_amplitude_spectrum_mask(dft):
    # 作傅里叶变换, 通过幅度谱将票据中的波纹去除.
    amplitude_spectrum = np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]) + 1e-10)
    amplitude_spectrum = np.array(amplitude_spectrum / amplitude_spectrum.max() * 255, dtype=np.uint8)
    show_image(amplitude_spectrum)

    result, binary = cv.threshold(amplitude_spectrum, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.erode(binary, kernel)
    return binary


def idft_transformation(dft, amplitude_mask):
    """
    :param dft:
    :param amplitude_mask: 幅度谱蒙版, 保留的地方为 1, 不保留的地方为 0.
    :return:
    """
    gray_spectrum = np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
    phase_spectrum = np.arctan2(dft[:, :, 1], dft[:, :, 0])

    spectral_residual = np.exp(gray_spectrum * amplitude_mask)
    cos_spectrum = np.expand_dims(np.cos(phase_spectrum) * spectral_residual, axis=2)
    sin_spectrum = np.expand_dims(np.sin(phase_spectrum) * spectral_residual, axis=2)

    new_dft = np.concatenate([cos_spectrum, sin_spectrum], axis=2)
    idft = cv.dft(new_dft, cv.DFT_INVERSE)

    idft_magnitude = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
    idft_magnitude = np.array(idft_magnitude / idft_magnitude.max() * 255, dtype=np.uint8)
    return idft_magnitude


if __name__ == '__main__':
    image_path = '../dataset/data/airport/airport1.jpg'
    image = cv.imread(image_path)

    approx = get_paper_corner(image)
    # cv.polylines(image, [approx], True, (0, 255, 0), 2)
    # show_image(image)

    image = perspective_transformation(image, approx)
    show_image(image)

    # 作傅里叶变换, 通过幅度谱将票据中的波纹去除.
    dft = dft_transformation(image)

    print(dft.shape)
    binary = get_amplitude_spectrum_mask(dft)
    binary[900: 1020, 660: 780] = 0
    binary[950: 970, :] = 0
    binary[:, 710: 730] = 0
    show_image(binary)
    amplitude_mask = np.where(binary == 0, 1, 0)

    result = idft_transformation(dft, amplitude_mask)
    show_image(result)
    # 去除波纹线失败.
