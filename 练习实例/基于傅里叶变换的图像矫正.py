import cv2 as cv
import numpy as np
import math


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def expand_image(image):
    h, w = image.shape[:2]
    new_h = cv.getOptimalDFTSize(h)
    new_w = cv.getOptimalDFTSize(w)
    right = new_w - w
    bottom = new_h - h
    result = cv.copyMakeBorder(image, 0, bottom, 0, right, borderType=cv.BORDER_CONSTANT, value=0)
    return result


def fourier_demo1():
    # 1. 灰度化读取文件，
    image = cv.imread('../dataset/data/exercise_image/image_text_r.png',0)

    # 2. 图像延扩
    # n_image = expand_image(image)

    # 3. 执行傅里叶变换中心化, 得到幅度谱图像. (我不知道, 为什么会有一条竖线).
    h, w = image.shape
    center_array = np.array([np.power(-1, r + c) for r in range(h) for c in range(w)]).reshape(image.shape)
    center_image = image * center_array
    dft = cv.dft(np.float32(center_image), flags=cv.DFT_COMPLEX_OUTPUT)

    dft_magnitude = np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
    dft_magnitude = np.array(dft_magnitude / dft_magnitude.max() * 255, dtype=np.uint8)
    show_image(dft_magnitude)

    # 二值化
    ret, binary = cv.threshold(dft_magnitude, 175, 255, cv.THRESH_BINARY)
    show_image(binary)

    # 霍夫直线变换
    lines = cv.HoughLinesP(binary, 2, np.pi/180, 30, minLineLength=40, maxLineGap=100)

    # 创建一个新图像，标注直线
    piThresh = np.pi/180
    pi2 = np.pi/2
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(dft_magnitude, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if x2 - x1 == 0:
            continue
        else:
            theta = (y2 - y1) / (x2 - x1)
        if abs(theta) < piThresh or abs(theta - pi2) < piThresh:
            continue

    angle = math.atan(theta)
    angle = angle * (180 / np.pi)
    angle = (angle - 90)/(w/h)

    center = (w//2, h//2)

    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    show_image(dft_magnitude)
    show_image(rotated)
    return


def fourier_demo2():
    # 1. 灰度化读取文件，
    image = cv.imread('../dataset/data/exercise_image/image_text_r.png', 0)
    show_image(image)
    h, w = image.shape

    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)

    dft_magnitude = np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
    dft_magnitude = np.array(dft_magnitude / dft_magnitude.max() * 255, dtype=np.uint8)
    show_image(dft_magnitude)

    # 二值化
    ret, binary = cv.threshold(dft_magnitude, 175, 255, cv.THRESH_BINARY)
    show_image(binary)

    # 霍夫直线变换
    lines = cv.HoughLinesP(binary, 2, np.pi / 180, 30, minLineLength=40, maxLineGap=100)

    # 创建一个新图像，标注直线
    piThresh = np.pi / 180
    pi2 = np.pi / 2
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(dft_magnitude, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if x2 - x1 == 0:
            continue
        else:
            theta = (y2 - y1) / (x2 - x1)
        if abs(theta) < piThresh or abs(theta - pi2) < piThresh:
            continue

    angle = math.atan(theta)
    angle = angle * (180 / np.pi)
    angle = (angle - 90) / (w / h)

    center = (w // 2, h // 2)

    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    show_image(dft_magnitude)
    show_image(rotated)
    return


if __name__ == '__main__':
    fourier_demo2()

