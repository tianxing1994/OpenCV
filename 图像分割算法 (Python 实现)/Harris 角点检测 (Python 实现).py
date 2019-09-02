"""
参考链接:
https://www.cnblogs.com/zyly/p/9508131.html

Harris 检测原理
Harris 认为, 当我们用一个的窗口放在图像上的 (x, y) 点时, 如果该点是一个角点,
则不论我们是延着 x 方向, 还是 y 方向移动窗口, 都应该会使窗口内的灰度值总和产生明显变化.

以上是 Harris 角点检测的出发思想, 但是在实际的推导过程中, 又得出了具体的实现方法:

我们计算窗口内各像素点在 x, y 方向上的梯度, 并进行统计. 得到样本集:
[[Ix1, Iy1],
[Ix2, Iy2],
...
[Ixn, Iyn]]
其中 Ixn, Iyn 分别代表窗口内各点的 x 方向梯度值和 y 方向上的梯度值.

以 Iy 和 Iy 做为特征, 画出 xoy 笛卡尔坐标系及样本的分布情况.
如果窗口在角点上, 则样本在 x, y 方向上都应分布得分开.
如果窗口在边缘上, 则样本在 x, y 方向上只会有一个方向上分布得较开, 另一个方向上分布得密集.
如果窗口在平坦区域, 则样本在 x, y 方向上都分布得密集.

有了以上的判断, 可以想到, 只要对窗口中的像素求 x, y 方向的梯度, 再判断这些梯度的分布情况. 就可以分别出角点, 边缘, 平坦区域.

那么, 用什么方法来量化呢 ?

PCA 主成分分析. (也是 PCA 降维的核心思想).
用 PCA 原理, 求协方差矩阵. 再求矩阵的 2 个特征值 λ1, λ2. 建立方程:
λ1 * λ2 - k * (λ1 + λ2)^2
其中 k 是经验值, 一般取 0.04-0.06 之间

以上方程变换为:
- (k*λ1^2 + (2k-1)*λ1*λ2 + k*λ2^2)

则不论在 x 还是 y 方向上都会得到相当数量且具有较大梯度值的点.

以上的实现, 详见参考链接.

Harries 角点的性质:
1. 增大 k 的值, 降低角点检测的灵敏度, 减少被检测角点的数量; 减少 k 值, 增加角点检测的灵敏度, 增加被检测角点的数量.
2. Harris 角点检测算子对亮度和对比度的变化不灵敏. 因为改变亮度和对比度基本不会影响梯度计算.
3. Harris 角点检测算子具有旋转不变性.
4. Harris 角点检测算子不具有尺度不变性. 因为当图像被放大缩小时, 在检测窗口尺寸不变的前提下, 在窗口内所包含图像的内容是完全不同的.

"""
# -*- coding: utf-8 -*-
import cv2
import numpy as np


def harris_detect(image, ksize=3):
    k = 0.04  # 响应函数k
    threshold = 0.05  # 设定阈值
    WITH_NMS = False  # 是否非极大值抑制

    # 1、使用Sobel计算像素点x,y方向的梯度
    h, w = image.shape[:2]
    # Sobel函数求完导数后会有负值, 还有会大于255的值. 而原图像是uint8，即8位无符号数, 所以Sobel建立的图像位数不够, 会有截断.
    # 因此要使用16位有符号的数据类型, 即 cv2.CV_16S.
    grad = np.zeros((h, w, 2), dtype=np.float32)
    grad[:, :, 0] = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
    grad[:, :, 1] = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)

    # 2、计算Ix^2,Iy^2,Ix*Iy
    m = np.zeros((h, w, 3), dtype=np.float32)
    m[:, :, 0] = grad[:, :, 0] ** 2
    m[:, :, 1] = grad[:, :, 1] ** 2
    m[:, :, 2] = grad[:, :, 0] * grad[:, :, 1]

    # 3、利用高斯函数对Ix^2,Iy^2,Ix*Iy进行滤波
    m[:, :, 0] = cv2.GaussianBlur(m[:, :, 0], ksize=(ksize, ksize), sigmaX=2)
    m[:, :, 1] = cv2.GaussianBlur(m[:, :, 1], ksize=(ksize, ksize), sigmaX=2)
    m[:, :, 2] = cv2.GaussianBlur(m[:, :, 2], ksize=(ksize, ksize), sigmaX=2)

    m = [np.array([[m[i, j, 0],
                    m[i, j, 2]],

                   [m[i, j, 2],
                    m[i, j, 1]]]) for i in range(h) for j in range(w)]

    # 4、计算局部特征结果矩阵M的特征值和响应函数R(i,j)=det(M)-k(trace(M))^2, 0.04<=k<=0.06. det(M)=λ1*λ2, trace(M)=λ1+λ2
    # 特征值乘积等于对应方阵行列式的值, 特征值的和等于对应方阵对角线元素之和.
    D, T = list(map(np.linalg.det, m)), list(map(np.trace, m))
    R = np.array([d - k * t ** 2 for d, t in zip(D, T)])

    # 5、将计算出响应函数的值R进行非极大值抑制, 滤除一些不是角点的点, 同时要满足大于设定的阈值
    # 获取最大的R值
    R_max = np.max(R)
    R = R.reshape(h, w)
    corner = np.zeros_like(R, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if WITH_NMS:
                # 除了进行进行阈值检测 还对3x3邻域内非极大值进行抑制(导致角点很小, 会看不清)
                if R[i, j] > R_max * threshold and R[i, j] == np.max(
                        R[max(0, i - 1):min(i + 2, h - 1), max(0, j - 1):min(j + 2, w - 1)]):
                    corner[i, j] = 255
            else:
                # 只进行阈值检测
                if R[i, j] > R_max * threshold:
                    corner[i, j] = 255
    return corner


if __name__ == '__main__':
    image = cv2.imread('C:/Users/Administrator/PycharmProjects/OpenCV/dataset/contours.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dst = harris_detect(gray)

    image[dst == 255] = (0, 0, 255)

    cv2.namedWindow("harris", cv2.WINDOW_NORMAL)
    cv2.imshow('harris', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

