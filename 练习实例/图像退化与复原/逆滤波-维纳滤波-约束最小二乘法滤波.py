# coding=utf8
"""
参考链接：
https://blog.csdn.net/wsp_1138886114/article/details/95024180
"""
import numpy as np
import cv2 as cv


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def flip180(src):
    ret = np.flip(src, axis=0)
    return ret


def get_motion_blur_kernel(dsize, angle=0):
    m = cv.getRotationMatrix2D((round(dsize / 2), round(dsize / 2)), angle, 1)
    diag = np.diag(np.ones(dsize))
    kernel = cv.warpAffine(diag, m, (dsize, dsize))
    ret = kernel / dsize
    return ret


def get_motion_blur_kernel2(motion_distance, motion_angle):
    """
    :param motion_distance: 成像中运动的距离
    :param motion_angle: 以度数输入的运动角度.
    x 轴向右, y 轴向下, 按 x 轴顺时针旋转的角度. 最大为 180 度.
    :return:
    """
    angle_radian = motion_angle * np.pi / 180.0
    slope_tan = np.tan(angle_radian)
    slope_cotan = 1.0 / slope_tan
    h = abs(int(round(motion_distance * np.sin(angle_radian)))) + 1
    w = abs(int(round(motion_distance * np.cos(angle_radian)))) + 1
    ret = np.zeros(shape=(h, w), dtype=np.float64)
    if 0 <= slope_tan <= 1:
        for i in range(w):
            x = i
            y = int(i * slope_tan)
            ret[y, x] = 1.0
        ret /= np.sum(ret)
        return ret
    elif slope_tan > 1:
        for i in range(h):
            y = i
            x = int(i * slope_cotan)
            ret[y, x] = 1.0
        ret /= np.sum(ret)
        return ret
    elif -1 <= slope_tan < 0:
        for i in range(w):
            x = i
            y = int(i * slope_tan) - 1
            ret[y, x] = 1.0
        ret /= np.sum(ret)
        return ret
    elif slope_tan < -1:
        for i in range(h):
            y = i
            x = int(i * slope_cotan) - 1
            ret[y, x] = 1.0
        ret /= np.sum(ret)
        return ret


def copy_make_border(sub_array, to_shape):
    s_h, s_w = sub_array.shape
    t_h, t_w = to_shape
    top = (t_h - s_h) // 2
    bottom = t_h - s_h - top
    left = (t_w - s_w) // 2
    right = (t_w - s_w - left)
    ret = cv.copyMakeBorder(sub_array,
                            top=top,
                            bottom=bottom,
                            left=left,
                            right=right,
                            borderType=cv.BORDER_CONSTANT,
                            value=0)
    return ret


def make_blurred(src, psf, eps=1e-3):
    input_fft = np.fft.fft2(src)
    psf_fft = np.fft.fft2(psf) + eps
    blurred_ = np.fft.fftshift(np.fft.ifft2(input_fft * psf_fft))
    blurred = np.array(np.abs(blurred_), dtype=np.uint8)
    return blurred, blurred_


def inverse_filter(blurred, psf, eps=1e-3):
    """
    逆滤波
    :param blurred: 模糊图像
    :param psf: 估计的模糊图像所用的卷积核.
    :param eps:
    :return:
    """
    input_fft = np.fft.fft2(blurred)
    psf_fft = np.fft.fft2(psf) + eps
    inverse_filter_deblurred_ = np.fft.fftshift(np.fft.ifft2(input_fft / psf_fft))
    inverse_filter_deblurred = np.array(np.abs(inverse_filter_deblurred_), dtype=np.uint8)
    return inverse_filter_deblurred, inverse_filter_deblurred_


def wiener(blurred, psf, eps=1e-3, k=1e-2):
    """
    维纳滤波.
    :param blurred: 模糊图像
    :param psf: 估计的模糊图像所用的卷积核.
    :param eps:
    :param k:
    :return:
    """
    input_fft = np.fft.fft2(blurred)
    psf_fft = np.fft.fft2(psf) + eps
    winner_fft = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + k)
    wiener_deblurred_ = np.fft.fftshift(np.fft.ifft2(input_fft * winner_fft))
    wiener_deblurred = np.array(np.abs(wiener_deblurred_), dtype=np.uint8)
    return wiener_deblurred, wiener_deblurred_


def clsf(blurred, psf, gamma=0.005):
    """
    约束最小二乘法滤波.
    :param blurred: 模糊图像
    :param psf: 估计的模糊图像所用的卷积核.
    :param gamma:
    :return:
    """
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    f_kernel = copy_make_border(kernel, to_shape=blurred.shape)
    pf = np.fft.fft2(psf)
    pf_kernel = np.fft.fft2(f_kernel)
    if_noisy = np.fft.fft2(blurred)
    numerator = np.conj(pf)
    denominator = pf ** 2 + gamma * (pf_kernel ** 2)
    clsf_deblurred_ = np.fft.fftshift(np.fft.ifft2(numerator * if_noisy / denominator))
    clsf_deblurred = np.array(np.real(clsf_deblurred_), dtype=np.uint8)
    return clsf_deblurred, clsf_deblurred_


def demo1():
    image_path = "../../dataset/data/airport/airport1.jpg"
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 运动模糊滤波核函数, 两种函数.
    # motion_blur_kernel = get_motion_blur_kernel2(motion_distance=10, motion_angle=60)
    motion_blur_kernel = get_motion_blur_kernel(dsize=15, angle=60)

    psf = copy_make_border(sub_array=motion_blur_kernel, to_shape=(h, w))

    # 频滤域滤波图像模糊
    blurred, blurred_ = make_blurred(gray, psf)
    show_image(blurred)

    # 逆滤图像复原
    # cleared, cleared_ = inverse_filter(blurred_, psf)
    # 维纳滤波图像复原
    # cleared, cleared_ = wiener(blurred_, psf, k=1e-20)
    # 约束最小二乘法滤波.
    cleared, cleared_ = clsf(blurred, psf)
    show_image(cleared)
    return


def demo2():
    image_path = "../../dataset/data/airport/airport1.jpg"
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 运动模糊滤波核函数, 两种函数.
    # motion_blur_kernel = get_motion_blur_kernel2(motion_distance=15, motion_angle=60)
    motion_blur_kernel = get_motion_blur_kernel(dsize=15, angle=60)

    # 使用卷积核将图像模糊
    motion_blur_kernel = np.flip(motion_blur_kernel, axis=0)
    blurred_ = cv.filter2D(gray, ddepth=cv.CV_32F, kernel=motion_blur_kernel)
    blurred = np.array(blurred_, dtype=np.uint8)
    show_image(blurred)

    psf = copy_make_border(sub_array=motion_blur_kernel, to_shape=(h, w))
    # 逆滤图像复原
    # cleared, cleared_ = inverse_filter(blurred_, psf)
    # 维纳滤波图像复原
    # cleared, cleared_ = wiener(blurred_, psf, k=1e-3)
    # 约束最小二乘法滤波.
    cleared, cleared_ = clsf(blurred, psf, gamma=0.0007)
    show_image(cleared)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
