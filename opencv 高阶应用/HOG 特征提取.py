# coding=utf-8
"""
参考链接:
https://www.cnblogs.com/alexme/p/11361563.html
https://github.com/udacity/CVND_Exercises/blob/master/1_4_Feature_Vectors/3_1.%20HOG.ipynb
https://www.cnblogs.com/lyx2018/p/7123794.html
https://blog.csdn.net/ppp8300885/article/details/71078555

Histograms of Oriented Gradients (HOG)
正如在 ORB 算法中看到的, 我们可以使用图像中的关键点进行匹配, 以检测图像中的对象.
当想要检测具有许多一致的内部特性且不受背景影响的对象时, 这些类型的算法非常有用. 例如, 这些算法在人脸检测中可以取得良好的效果,
因为人脸有许多不受图像背景影响的一致的内部特征, 例如眼睛, 鼻子和嘴巴.
然而, 当试图进行更一般的对象识别时, 例如图像中的行人检测时,
这些类型的算法并不能很好地工作. 原因是人们的内在征不像脸那样一致, 因为每个人的体型和风格都不同.
这意味着每个人都会有一套不同的内部特征, 因此我们需要一些能够更全面地描述一个人的东西.

一种选择是尝试通过行人的轮廓来检测它们. 通过图像的轮廓(边界)来检测物体是非常具有挑战性的,
因为我们必须处理背景和前景之间的对比带来的困难. 例如, 假设想检测的一个图像中的行人,
她正走在一栋白色建筑前, 穿着白色外套和黑色裤子. 我们可以在下图中看到, 由于图像的背景大多是白色, 黑色裤子的对比度将非常高,
但由于外套也是白色的, 所以对比度将非常低.

在这种情况下, 检测裤子的边缘是很容易的, 但是检测外套的边缘是非常困难的. 而这就是为什么需要 HOG. 即 "定向梯度柱状图",
它是由 Navneet Dalal 和 Bill Triggs 于 2005 年首次引入的.

HOG 算法的工作原理是创建图像中梯度方向分布的柱状图, 然后以一种非常特殊的方式对其进行归一化.
这种特殊的归一化使得 HOG 能够有效地检测物体的边缘, 即使在对比度很低的情况下也是如此.
这些标准化的柱状图被放在一个特征向量 (称为 HOG 描述符) 中, 可以用来训练机器学习算法, 例如支持向量机 (SVM),
以根据图像中的边界 (边) 检测对象. 由于它的巨大成功和可靠性, HOG 已成为计算机视觉中应用最广泛的目标检测算法之一.


### HOG 算法
顾名思义, HOG 算法基于从图像梯度方向创建直方图. HOG 算法通过以下一系列步骤实现.
1. 给定特定对象的图像, 设置一个覆盖图像中整个对象的检测窗口 (感兴趣区域).
2. 计算检测窗口中每个像素的梯度大小和方向.
3. 将检测窗口分成像素的连接单元格, 所有单元格的大小相同. 单元格的大小是一个自由参数, 通常选择它来匹配要检测的特征的比例.
例如, 在一个 64*128 像素的检测窗口中, 6 到 8 像素宽的方形单元格适用于检测人体肢体.
4. 为每个单元创建一个柱状图, 首先将每个单元中所有像素的渐变方向分组为特定数量的方向 (角度) 箱,
然后将每个角度箱中渐变的渐变幅度相加. 柱状图中的箱数是一个自由参数, 通常设置为 9 个角箱.
5. 将相邻单元分组成块. 每个块中的单元格数是一个自由参数, 所有块的大小都必须相同. 每个块之间的距离 (称为跨距) 是一个自由参数,
但它通常设置为块大小的一半, 在这种情况下, 将得到重叠块. 经验表明, 该算法能更好地处理重叠块.
6. 使用每个块中包含的单元格来规范化该块中的单元格柱状图. 如果有重叠块, 这意味着大多数单元格将针对不同的块进行规格化. 因此,
同一个单元可能有不同的归一化.
7. 将所有块中的所有标准化柱状图收集到一个称为 HOG 描述符的特征向量中.
8. 使用从包含同一对象的许多图像中得到的 HOG 描述符训练机器学习算法, 例如使用 SVM, 以检测图像中的这些对象. 例如,
可以使用来自许多行人图像的 HOG 描述符来训练 SVM 以检测图像中的行人. 训练通过使用包含目标的正例和不包含目标的负例完成.
9. 一旦对支持向量机进行了训练, 就命名用滑动窗口方法来尝试检测和定位图像的对象.
检测图像中的对象需要找到图像中与 SVM 学习到的 HOG 模式相似的部分.


### 为什么 HOG 算法有效
正如我们上面所了解的, HOG 通过在图像的局部区域中添加特定方向的梯度大小来创建柱状图, 称为 "cells",
通过这样做可以保证更强的梯度对它们各自的角度柱状图的大小贡献更大, 同时最小化由噪声引起的弱梯度和随机定向梯度的影响.
通过这种方式, 柱状图告诉我们每个单元格的主要梯度方向.

#### 处理相似性问题
现在考虑一个问题, 由于局部照明变化以及背景和前景之间的对比度, 梯度方向的大小可以有很大的变化.
为了考虑背景-前景对比度的差异, HOG 算法法试在局部检测边缘. 为了做到这一点, 它定义了称为块的单元格组,
并使用该局部单元格组规范化柱状图. 通过局部归一化, HOG 算法可以非常可靠地检测每个块中的边缘, 这称为块归一化.

除了使用块规范化之外, HOG 算法还使用重叠块来提高其性能. 通过使用重叠块, 每个单元为最终的 HOG 描述符提供几个独立的组成部分,
其中每个组成部分对应于一个针对不两只块进行规范化的单元. 这似乎是多余的, 但是经验表明, 通过对每个单元对不同局部块进行多次规格化,
HOG 算法的性能显著提高.

### 使用 OpenCV 的 HOGDescriptor 类
使用 OpenCV 的 HOGDescriptor 类来创建 HOG 描述符. HOG 描述符的参数是使用 HOGDescriptor() 函数设置的.
常用参数为前几项: win_size, block_size, block_stride, cell_size 和 nbins, 其他参数一般可以保留其默认值.
:param win_size: 像素级, 检测窗口的大小 (width, height). 定义感兴趣区域. 其必须是 cell_size 的整数倍.
:param block_size: 像素级, 块大小 (width, height). 定义每一个 block 块中有多少个 cell.
必须是 cell_size 的整数倍, 同时必须比检测窗口小. block 块越小, 你可以获得的细节越细.
:param block_stride: 像素级, 块的步幅 (horizontal, vertical). 其必须是 cell_size 的整数倍.
block_stride 定义相邻块之间的距离, 例如: 水平方向 8 个像素, 竖直方向 8 个像素.
越大的 block_strides 步幅使得算法运行更快 (因为更少的块需要计算) 但算法表现可能不好.
:param cell_size: 像素级, 单元的大小 (width, height). 定义 cell 单元的大小. 越小的 cell_size 则越细致的细节会被检测.
:param nbins: bins 指定直方向的箱数. 确定用于制作直方图的角度仓的数量.
越多的箱数则代表更多的梯度方向. HOG 使用无符号梯度, 因此角度单元的值将介于 0-180 之间.
:param win_sigma: 高斯平滑窗口的参数. 在计算梯度直方图之前先对每个 block 块边缘附近的像素应用高斯平滑, 可以提高 HOG 算法的效果.
:param threshold_L2hys: L2-Hys (Lowe 风格的 L2 范数裁剪) 归一化收缩方法.
L2-Hys 方法用于对块进行规范化, 它由 L2 范数, 裁剪和重新规范化组成.
裁剪, 以使得描述符向量的最大值不大于给定阈值 (默认为 0.2), 即大于阈值的取最大值(阈值).
裁剪后, 描述符向量按照 IJCV, 60(2):91-110, 2004 中所述进行重新规范化.
:param gamma_correction: 用于指定是否需要伽马校正预处理的标志. 执行伽马校正会稍微提高 HOG 算法的性能.
:param nlevel: 最大检测窗口增加.

### HOG 描述符中的元素数量
HOG 描述符(特征向量) 是由检测窗口中的所有块的所有单元的归一化直方图 concat 起来的长向量.
因此, HOG 特征向量的大小将由检测窗口中的块总数乘以每个块的单元数乘以定向箱 (bin) 的数量来给出:
total_elements = total_number_of_blocks * number_cells_per_block * number_of_bins

如果我们没有重叠块 (即: block_stride 等于 block_size 的情况), 则可以通过将检测窗口的大小除以块大小来容易地计算块的总数.
但是, 在一般情况下, 我们必须考虑到有重叠块的事实, 要查找一般情况下的块总数 (即对任何 block_size 和 block_size),
我们可以使用下面给出的公式:
block_total = (block_size / block_stride) * (window_size / block_size) - ((block_size / block_stride) - 1); for i = x, y
其中的 size, stride 分别表示 x, y 方向的, 也就是只计算 window 中 x 或 y 一个方向上的 block 的数量.
上述公式化简为:
block_total = (cells - num_cells_per_block) / n + 1; for i = x, y
其中 cells 是沿检测窗口 x 或 y 方向的单元格总数. n 是以 cell_size 为单位的 x 或 y 方向的块步幅.

以上内容可能不好理解, 其实:
一个大的窗口为 window_size, 小窗口为 block_size, 移动的步幅为 block_stride.
我们很容易计算出, 在一个方向上小窗口能够移动多少次:
(window_size - blocksize) / block_stride
可以移动到新位置的次数加上本来的位置, 则在一个方向上的 block 数量为:
(window_size - blocksize) / block_stride + 1

### 可视化 HOG 描述符
我们可以通过将与每个 cell 单元相关联的直方图绘制为矢量集合来可视化 HOG 描述符.
为此, 我们将直方图中的每个 bin 绘制为单个向量, 其大小由 bin 的高度给出, 其方向由与其关联的角度 bin 给出.
由于任何给定的单元格可能有多个与之关联的直方图, 因为存在重叠的块,
我们将选择平均每个单元格的所有直方图, 以便为每个单元格生成单个直方图.
"""
import math

import numpy as np
import cv2 as cv


def show_image(image, win_name='input image'):
    # cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def hog_hist2image(hog_hist, image_size, win_size, win_stride, padding, block_size, block_stride, cell_size, nbins):
    """cv2.HOGDescriptor 返回的 HOG 特征直方图可视化"""

    # 1. 每个 cell 中有 nbins 个值.
    # 2. 每个 block 中有 (m, n) 个 cell
    m = int(block_size[0] / cell_size[0])
    n = int(block_size[1] / cell_size[1])

    # 3. 每个 window 中有 (a, b) 个 block
    a = int((win_size[0] - block_size[0]) // block_stride[0] + 1)
    b = int((win_size[1] - block_size[1]) // block_stride[1] + 1)

    # 4. 本图片中有 (x, y) 个 window
    x = int((image_size[0] + 2 * padding[0] - win_size[0]) // win_stride[0] + 1)
    y = int((image_size[1] + 2 * padding[1] - win_size[1]) // win_stride[1] + 1)

    gradient_data = np.reshape(hog_hist, newshape=(y, x, b, a, n, m, nbins))
    print(gradient_data.shape)

    # 本图片应有 (i, j) 个 cell.
    i = int((image_size[0] + 2 * padding[0]) / cell_size[0])
    j = int((image_size[1] + 2 * padding[1]) / cell_size[1])
    print(j, i, nbins)

    # 梯度累加, 累加计数.
    gradient_accumulate = np.zeros(shape=(j, i, nbins))
    gradient_count = np.zeros(shape=(j, i, 1))
    for y_ in range(y):
        for x_ in range(x):
            for b_ in range(b):
                for a_ in range(a):
                    j_ = int((y_*win_stride[1]+b_*block_stride[1]) / cell_size[1])
                    i_ = int((x_*win_stride[0]+a_*block_stride[0]) / cell_size[0])
                    gradient_accumulate[j_: j_ + m, i_: i_ + n] += gradient_data[y_, x_, b_, a_]
                    gradient_count[j_: j_ + m, i_: i_ + n] += 1

    cell_gradient = gradient_accumulate / gradient_count
    hog_image = np.zeros(shape=(image_size[1] + padding[1], image_size[0] + padding[0]))
    angle_unit = 180 / nbins
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
                x = int(i * cell_size[0] + cell_size[0] / 2)
                y = int(j * cell_size[1] + cell_size[1] / 2)
                # 图像中, 该线的长度应在 cell 范围内. 大小为: lengh = magnitude * cell_size
                # lengh = magnitude * cellSize[0]
                lengh = magnitude
                # 根据该梯度值所属的方向. 求取 x1, y1, x2, y2
                x1 = int(x + (lengh / 2) * math.cos(angle_radian))
                y1 = int(y + (lengh / 2) * math.sin(angle_radian))
                x2 = int(x - (lengh / 2) * math.cos(angle_radian))
                y2 = int(y - (lengh / 2) * math.sin(angle_radian))
                # 画线, 线的颜色深浅由梯度值的大小决定.
                cv.line(hog_image, (y1, x1), (y2, x2), color=int(255 * math.sqrt(magnitude)), thickness=1)
                # 下一个梯度值的角度方向为 angle:
                angle += angle_gap

    result = np.array(hog_image, dtype=np.uint8)
    return result


def calc_els_number(win_size, block_size, block_stride, num_cells_per_block, nbins):
    # 通过 win_size, blocksize, block_stride 参数计算 block 的数量, 及 hog_descriptor 的维数.
    block_total = ((win_size[0] - block_size[0]) // block_stride[0] + 1) * \
    ((win_size[1] - block_size[1]) // block_stride[1] + 1)

    total_eis = block_total * num_cells_per_block[0] * num_cells_per_block[1] * nbins
    print(total_eis)
    return


def demo1():
    """调用 cv2.HOGDescriptor 计算图像 HOG 描述符. 计算描述符的数量. """
    image_path = '../dataset/data/other_sample/triangle_tile.jpeg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    cell_size = (6, 6)
    num_cells_per_block = (2, 2)
    block_size = (cell_size[0] * num_cells_per_block[0],
    cell_size[1] * num_cells_per_block[1])
    # 计算当前图像中 x 和 y 方向上能应用的 cell 单元个数. 图像中多余的部分舍弃掉.
    x_cells = gray.shape[1] // cell_size[0]
    y_cells = gray.shape[0] // cell_size[1]
    # 在 x, y 方向上, block 窗口按 cell 单元移动的步幅大小.
    # 必须为整数, 且要满足:
    # (x_cells - num_cells_per_block[0]) / x_stride = integer
    # (y_cells - num_cells_per_block[1]) / y_stride = integer
    x_stride = 1
    y_stride = 1
    # 像素级边的 block 步幅 (horizantal, vertical). 其必须是 cell_size 的整数倍.
    block_stride = (cell_size[0] * x_stride, cell_size[1] * y_stride)
    # 梯度方向的箱数.
    nbins = 9
    # 检测窗口的大小, 这里取窗口大小与图像一样大.
    win_size = (x_cells * cell_size[0], y_cells * cell_size[1])
    # 接收位置参数.
    args = (win_size, block_size, block_stride, cell_size, nbins)
    hog = cv.HOGDescriptor(*args)
    # 在灰度图像上计算 HOG 描述符.
    hog_descriptor = hog.compute(gray)
    print(hog_descriptor.shape)
    calc_els_number(win_size, block_size, block_stride, num_cells_per_block, nbins)
    return


def demo2():
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
    L2HysThreshold = 2e-01
    gammaCorrection = 0
    nlevels = 64

    # 定义对象hog，同时输入定义的参数， 剩下的默认即可
    # hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
    #                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    # 这个函数是个奇葩, 接收位置参数, 不能写作 winSize=winSize 的关键字参数.
    args = (winSize, blockSize, blockStride, cellSize, nbins,
            derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    hog = cv.HOGDescriptor(*args)

    result = hog.compute(image, winStride=(16, 16), padding=(0, 0))
    print(result.shape)
    return


def demo3():
    image = cv.imread('../dataset/data/other_sample/triangle_tile.jpeg', 0)
    image = cv.resize(image, dsize=(246, 246))
    show_image(image)
    print(image.shape)
    image_size = image.shape[::-1]
    win_size = (246, 246)
    block_size = (12, 12)
    block_stride = (6, 6)
    cell_size = (6, 6)
    nbins = 9

    deriv_aperture = 1
    win_sigma = 4.
    histogram_norm_type = 0
    threshold_L2hys = 2e-01
    gamma_correction = 0
    nlevels = 64

    win_stride = (16, 16)
    padding = (0, 0)

    # 这个函数是个奇葩, 接收位置参数, 不能写作 winSize=winSize 的关键字参数.
    # args = (win_size, block_size, block_stride, cell_size, nbins)
    args = (win_size, block_size, block_stride, cell_size, nbins,
            deriv_aperture, win_sigma, histogram_norm_type, threshold_L2hys, gamma_correction, nlevels)
    hog = cv.HOGDescriptor(*args)

    hog_hist = hog.compute(image, winStride=win_stride, padding=padding)
    print(hog_hist.shape)

    # hog_hist HOG 特征直方图可视化.
    result = hog_hist2image(hog_hist, image_size, win_size, win_stride, padding, block_size, block_stride, cell_size, nbins)
    show_image(result)
    return


if __name__ == '__main__':
    demo3()
