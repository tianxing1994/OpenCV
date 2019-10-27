# coding=utf-8
"""
不知道为什么跟 opencv 的不一样.

参考链接:
https://blog.csdn.net/liu111111113/article/details/81017896
https://blog.csdn.net/qq_36810544/article/details/83855962
RGB 无法直接转换成 LAB, 需要先转成 XYZ 再转换成 LAB, 即: RGB - XYZ - LAB

因此, 转换分为两部分.
第一步: RGB - XYZ 先对原图像进行对比度调整. 将 0-255 之间的 RGB 色彩值除以 255 归一化. 得到 x.
将 x 应用于以下公式, 得到新的 R, G, B 值.
gamma(x) = ((x + 0.055)/1.055)^2.4 if (x > 0.04045) else x/12.92
即: [R, G, B] = gamma([r, g, b] / 255)
存在转换矩阵 M:
0.4124, 0.3576, 0.1805
0.2126, 0.7152, 0.0722
0.0193, 0.1192, 0.9505

使得: [X, Y, X].T = M · [R, G, B].T

第二步:
XYZ - LAB f(t) = t^(1/3) if (t>(6/29)^3) else (1/3 * (29/6)^2 * t + 4/29)
其中:
(6/29)**3 = 0.008856
1/3 * (29/6)**2 = 7.787
4/29 = 0.137931

所以上式可以写作:
f(t) = np.power(t, 1/3) if t > 0.008856 else 7.787 * t + 0.137931
L* = 116 f(Y/Y_{n}) - 16
a* = 500 [f(X/X_{n}) - f(Y/Y_{n})]
b* = 200 [f(Y/Y_{n}) - f(Z/Z_{n})]
其中, L*, a*, b* 是最终的 LAB 色采空间三个通道的值.
X, Y, Z 是 RGB 转 XYZ 后计算出来的值, X_{n}, Y_{n}, Z_{n}
一般默认是 95.047, 100.0, 108.883.
"""
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def bgr2rgb(bgr):
    rgb = bgr[:, :, ::-1]
    return rgb


def gamma(x):
    x = x / 255
    gamma_0 = np.power((x + 0.055) / 1.055, 2.4)
    gamma_1 = x / 12.92
    mask_0 = np.where(x > 0.04045, 1, 0)
    mask_1 = np.where(x > 0.04045, 0, 1)
    result = gamma_0 * mask_0 + gamma_1 + mask_1
    return result


def rgb2xyz(rgb):
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    rgb = gamma(rgb)
    result = np.dot(rgb, m.T)
    return result

def f2(x):
    c_0 = np.power(x, 1/3)
    c_1 = 1/3 * np.power(29/6, 2) * x + 4/29
    mask_0 = np.where(x > np.power(6/29, 3), 1, 0)
    mask_1 = np.where(x > np.power(6/29, 3), 0, 1)
    result = c_0 * mask_0 + c_1 * mask_1
    return result


def f(t):
    t_0 = np.power(t, 1/3)
    t_1 = 7.787 * t + 0.137931
    mask_0 = np.where(t > 0.008856, 1, 0)
    mask_1 = np.where(t > 0.008856, 0, 1)
    result = t_0 * mask_0 + t_1 * mask_1
    return result


def xyz2lab(xyz):
    x = xyz[:, :, 0]
    y = xyz[:, :, 1]
    z = xyz[:, :, 2]
    f_x = f(x / 95.047)
    f_y = f(y / 100.0)
    f_z = f(z / 108.883)

    # 还有一种是当 y < 0.008856 时, l_ = 903.3 * f_y
    l_ = 116 * f_y - 16
    a_ = 500 * (f_x - f_y)
    b_ = 200 * (f_y - f_z)
    result = np.stack([l_, a_, b_], axis=2)
    return result


def demo1():
    image_path = "../../dataset/data/image_sample/bird.jpg"
    image = cv.imread(image_path)
    rgb = bgr2rgb(image)
    xyz = rgb2xyz(rgb)
    lab = xyz2lab(xyz)
    print(lab)
    print(lab.shape)
    return


def demo2():
    image_path = "../../dataset/data/image_sample/bird.jpg"
    image = cv.imread(image_path)
    lab_cv = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    show_image(lab_cv)
    print(lab_cv)
    return


if __name__ == "__main__":
    demo1()



