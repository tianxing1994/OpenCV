"""
参考链接：
https://www.cnblogs.com/alexme/p/11361563.html
https://www.cnblogs.com/lyx2018/p/7123794.html
https://blog.csdn.net/ppp8300885/article/details/71078555
"""
import math

import numpy as np
import cv2 as cv


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    # cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    """
    cv2.HOGDescriptor(), hog.compute() 调用.
    位置参数. 图像中装了窗口, 窗口中装了 block, block 中装了 cell.
    一个 cell 有 8 个向量值.
    一个 block 有 16 * 8 个向量值
    一个 window (64, 128),
    横向有 i = (winSize[0] - blockSize[0]) // blockStride[0] + 1 = (64 - 64) // 16 + 1 = 1 个 block.
    竖向有 j = (winSize[1] - blockSize[1]) // blockStride[1] + 1 = (128 - 64) // 16 + 1 = 5 个 block.
    则共有 i*j = 1*5 = 5 个 block, 有 5 * 16 * 8 = 640 个向量值.
    一张图片是 shape = (128, 128)
    横向有 a = (shape[0] - winSize[0]) // winStride[0] + 1 = (128 - 64) // 16 + 1 = 5 个 window.
    竖向有 b = (shape[1] - winSize[1]) // winStride[1] + 1 = (128 - 128) // 16 + 1 = 1 个 window.
    则共有 a*b = 5*1 = 5 个 window, 有 5 * 640 = 3200 个向量值.
    :return:
    """
    image = cv.imread('../dataset/data/other/silver.jpg', 0)
    image = cv.resize(image, dsize=(128, 128))
    print(image.shape)

    winSize = (64, 128)
    blockSize = (64, 64)
    blockStride = (16, 16)
    cellSize = (16, 16)
    nbins = 8
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64

    # 定义对象hog，同时输入定义的参数， 剩下的默认即可
    # hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
    #                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    # 这个函数是个奇葩, 接收位置参数, 不能写作 winSize=winSize 的关键字参数.
    args = (
    winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold,
    gammaCorrection, nlevels)
    hog = cv.HOGDescriptor(*args)

    result = hog.compute(image, winStride=(16, 16), padding=(0, 0))
    print(result.shape)
    return


def demo2():
    image = cv.imread('../dataset/data/other/silver.jpg', 0)
    image = cv.resize(image, dsize=(1024, 2048))
    show_image(image)
    print(image.shape)

    imageShape = image.shape[::-1]
    winSize = (64, 128)
    blockSize = (64, 64)
    blockStride = (16, 16)
    cellSize = (16, 16)
    nbins = 8
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01     # 0.2
    gammaCorrection = 0
    nlevels = 64

    winStride = (16, 16)
    padding = (0, 0)

    # 定义对象hog，同时输入定义的参数， 剩下的默认即可
    # hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
    #                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    # 这个函数是个奇葩, 接收位置参数, 不能写作 winSize=winSize 的关键字参数.
    args = (winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold,
    gammaCorrection, nlevels)
    hog = cv.HOGDescriptor(*args)

    result = hog.compute(image, winStride=winStride, padding=padding)
    print(result.shape)

    # 1. 每个 cell 中有 nbins 个值.
    # 2. 每个 block 中有 (m, n) 个 cell
    m = int(blockSize[0] / cellSize[0])
    n = int(blockSize[1] / cellSize[1])

    # 3. 每个 window 中有 (a, b) 个 block
    a = int((winSize[0] - blockSize[0]) // blockStride[0] + 1)
    b = int((winSize[1] - blockSize[1]) // blockStride[1] + 1)

    # 4. 本图片中有 (x, y) 个 window
    x = int((imageShape[0] + 2 * padding[0] - winSize[0]) // winStride[0] + 1)
    y = int((imageShape[1] + 2 * padding[1] - winSize[1]) // winStride[1] + 1)

    # 总共有 45 个窗口, 每个窗口包含 5 个 block, 每个 block 16 个 cell, 每个 cell nbins 个值.
    gradient_data = np.reshape(result, newshape=(y, x, b, a, n, m, nbins))
    print(gradient_data.shape)

    # 本图片应有 (i, j) 个 cell. i=8, j=16
    i = int((imageShape[0] + 2 * padding[0]) / cellSize[0])
    j = int((imageShape[1] + 2 * padding[1]) / cellSize[1])
    print(j, i, nbins)
    # 梯度累加, 累加计数.
    gradient_accumulate = np.zeros(shape=(j, i, nbins))
    gradient_weight = np.zeros(shape=(j, i, nbins))

    for y_ in range(y):
        for x_ in range(x):
            for b_ in range(b):
                for a_ in range(a):
                    j_ = int((y_*winStride[1]+b_*blockStride[1]) / cellSize[1])
                    i_ = int((x_*winStride[0]+a_*blockStride[0]) / cellSize[0])
                    # print(gradient_data[y_, x_, b_, a_].shape)
                    # print(j_, i_)
                    # print(gradient_accumulate[j_: j_ + m, i_: i_ + n].shape)
                    gradient_accumulate[j_: j_ + m, i_: i_ + n] += gradient_data[y_, x_, b_, a_]
                    gradient_weight[j_: j_ + m, i_: i_ + n] += 1
    cell_gradient = gradient_accumulate / gradient_weight
    # print(cell_gradient.shape)

    hog_image = np.zeros(shape=(imageShape[1] + padding[1], imageShape[0] + padding[0]))
    angle_unit = 360 / nbins
    for i in range(cell_gradient.shape[0]):
        for j in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[i][j]
            angle = 0
            angle_gap = angle_unit
            magnitude_max = np.max(cell_grad) + 1e-10
            for magnitude in cell_grad:
                magnitude = magnitude / magnitude_max
                # 角度转弧度
                angle_radian = math.radians(angle)
                # 在图像中每个 cell 的范围内画线, 以表达该 cell 梯度的方向与大小.
                # 求出 cell 的中心点坐标.
                x = int(i * cellSize[0] + cellSize[0] / 2)
                y = int(j * cellSize[1] + cellSize[1] / 2)
                # 图像中, 该线的长度应在 cell 范围内. 大小为: lengh = magnitude * cell_size
                lengh = magnitude * cellSize[0] * 10
                # 根据该梯度值所属的方向. 求取 x1, y1, x2, y2
                print(x, y)
                print(lengh)
                x1 = int(x + (lengh / 2) * math.cos(angle_radian))
                y1 = int(y + (lengh / 2) * math.sin(angle_radian))
                x2 = int(x)
                y2 = int(y)
                # x2 = int(x - (lengh / 2) * math.cos(angle_radian))
                # y2 = int(y - (lengh / 2) * math.sin(angle_radian))
                # 画线, 线的颜色深浅由梯度值的大小决定.
                cv.line(hog_image, (y1, x1), (y2, x2), color=int(255 * math.sqrt(magnitude)))
                # 下一个梯度值的角度方向为 angle:
                angle += angle_gap
    result = np.array(hog_image, dtype=np.uint8)
    show_image(result)
    return


def demo3():
    image = cv.imread('../dataset/data/other/head.jpg', 0)
    image = cv.resize(image, dsize=(1024, 2048))
    show_image(image)
    print(image.shape)

    imageShape = image.shape[::-1]
    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 8
    # derivAperture = 1
    # winSigma = 4.
    # histogramNormType = 0
    # L2HysThreshold = 2.0000000000000001e-01     # 0.2
    # gammaCorrection = 0
    # nlevels = 64

    winStride = (8, 8)
    padding = (0, 0)

    # 定义对象hog，同时输入定义的参数， 剩下的默认即可
    # hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
    #                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    # 这个函数是个奇葩, 接收位置参数, 不能写作 winSize=winSize 的关键字参数.
    # args = (winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
    # histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    args = (winSize, blockSize, blockStride, cellSize, nbins)
    hog = cv.HOGDescriptor(*args)

    result = hog.compute(image, winStride=winStride, padding=padding)
    print(result.shape)

    # 1. 每个 cell 中有 nbins 个值.
    # 2. 每个 block 中有 (m, n) 个 cell
    m = int(blockSize[0] / cellSize[0])
    n = int(blockSize[1] / cellSize[1])

    # 3. 每个 window 中有 (a, b) 个 block
    a = int((winSize[0] - blockSize[0]) // blockStride[0] + 1)
    b = int((winSize[1] - blockSize[1]) // blockStride[1] + 1)

    # 4. 本图片中有 (x, y) 个 window
    x = int((imageShape[0] + 2 * padding[0] - winSize[0]) // winStride[0] + 1)
    y = int((imageShape[1] + 2 * padding[1] - winSize[1]) // winStride[1] + 1)

    # 总共有 45 个窗口, 每个窗口包含 5 个 block, 每个 block 16 个 cell, 每个 cell nbins 个值.
    gradient_data = np.reshape(result, newshape=(y, x, b, a, n, m, nbins))
    print(gradient_data.shape)

    # 本图片应有 (i, j) 个 cell. i=8, j=16
    i = int((imageShape[0] + 2 * padding[0]) / cellSize[0])
    j = int((imageShape[1] + 2 * padding[1]) / cellSize[1])
    print(j, i, nbins)
    # 梯度累加, 累加计数.
    gradient_accumulate = np.zeros(shape=(j, i, nbins))
    gradient_weight = np.zeros(shape=(j, i, nbins))

    for y_ in range(y):
        for x_ in range(x):
            for b_ in range(b):
                for a_ in range(a):
                    j_ = int((y_*winStride[1]+b_*blockStride[1]) / cellSize[1])
                    i_ = int((x_*winStride[0]+a_*blockStride[0]) / cellSize[0])
                    # print(gradient_data[y_, x_, b_, a_].shape)
                    # print(j_, i_)
                    # print(gradient_accumulate[j_: j_ + m, i_: i_ + n].shape)
                    gradient_accumulate[j_: j_ + m, i_: i_ + n] += gradient_data[y_, x_, b_, a_]
                    gradient_weight[j_: j_ + m, i_: i_ + n] += 1
    cell_gradient = gradient_accumulate / gradient_weight
    # print(cell_gradient.shape)

    hog_image = np.zeros(shape=(imageShape[1] + padding[1], imageShape[0] + padding[0]))
    angle_unit = 360 / nbins
    for i in range(cell_gradient.shape[0]):
        for j in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[i][j]
            angle = 0
            angle_gap = angle_unit
            magnitude_max = np.max(cell_grad) + 1e-10
            for magnitude in cell_grad:
                magnitude = magnitude / magnitude_max
                # 角度转弧度
                angle_radian = math.radians(angle)
                # 在图像中每个 cell 的范围内画线, 以表达该 cell 梯度的方向与大小.
                # 求出 cell 的中心点坐标.
                x = int(i * cellSize[0] + cellSize[0] / 2)
                y = int(j * cellSize[1] + cellSize[1] / 2)
                # 图像中, 该线的长度应在 cell 范围内. 大小为: lengh = magnitude * cell_size
                lengh = magnitude * cellSize[0]
                # 根据该梯度值所属的方向. 求取 x1, y1, x2, y2
                # print(x, y)
                # print(lengh)
                x1 = int(x + (lengh / 2) * math.cos(angle_radian))
                y1 = int(y + (lengh / 2) * math.sin(angle_radian))
                x2 = int(x)
                y2 = int(y)
                # x2 = int(x - (lengh / 2) * math.cos(angle_radian))
                # y2 = int(y - (lengh / 2) * math.sin(angle_radian))
                # 画线, 线的颜色深浅由梯度值的大小决定.
                cv.line(hog_image, (y1, x1), (y2, x2), color=int(255 * math.sqrt(magnitude)))
                # 下一个梯度值的角度方向为 angle:
                angle += angle_gap
    result = np.array(hog_image, dtype=np.uint8)
    show_image(result)
    return


if __name__ == '__main__':
    demo3()
