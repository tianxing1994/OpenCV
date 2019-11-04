"""
参考链接:
https://blog.csdn.net/ppp8300885/article/details/71078555
https://www.cnblogs.com/alexme/p/11361563.html

HOG特征提取算法的整个实现过程大致如下：

1. 读入所需要的检测目标即输入的image
2. 将图像进行灰度化（将输入的彩色的图像的r,g,b值通过特定公式转换为灰度值）
3. 采用Gamma校正法对输入图像进行颜色空间的标准化（归一化）
4. 计算图像每个像素的梯度（包括大小和方向），捕获轮廓信息
5. 统计每个cell的梯度直方图（不同梯度的个数），形成每个cell的descriptor
6. 将每几个cell组成一个block（以2*2为例），一个block内所有cell的特征串联起来得到该block的HOG特征descriptor
7. 将图像image内所有block的HOG特征descriptor串联起来得到该image（检测目标）的HOG特征descriptor，这就是最终分类的特征向量
"""
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt


def show_image(image):
    cv.namedWindow('input image', cv.WINDOW_NORMAL)
    cv.imshow('input image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def get_gradient(image):
    """
    求取图像每个像素点的梯度, 与梯度方向.
    :param image:
    :return:
    """
    image = np.sqrt(image / float(np.max(image)))
    # show_image(image)
    gradient_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
    gradient_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)
    gradient_angle = cv.phase(gradient_x, gradient_y, angleInDegrees=True)
    return abs(gradient_magnitude), gradient_angle


def get_cell_gradient(cell_magnitude, cell_angle, bin_size):
    """
    按 cell 中各像素在各方向上的梯度大小之和. 考虑到每个像素的梯度大小, 按梯度方向在具体的角度上的分量.
    :param cell_magnitude: cell 中各像素的梯度大小.
    :param cell_angle: cell 中各像素的梯度方向.
    :return: 列表. 当前 cell 按梯度方向的梯度直方图.
    """
    # 将 360 度分成 bin_size 份, 求取 cell 内各像素的梯度方向直方图
    angle_unit = 180 / bin_size
    orientation_centers = [0] * bin_size
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            # 求 cell 内, 每个像素的梯度与梯度方向.
            gradient_strength = cell_magnitude[k, l]
            gradient_angle = cell_angle[k, l]

            # 求取当前梯度方向所属的角度范围, 比如: 分成 8 个方向时, angle_unit=45°,
            # 60° 则介于 1-2 之间. min_angle, max_angle = 1, 2
            min_angle = int(gradient_angle / angle_unit) % bin_size
            max_angle = (min_angle + 1) % bin_size

            # 求梯度方向在其方向单元内的角度, 如: 60° 介于 1-2 之间, 其在 1-2 之间的角度为 15°.
            mod = gradient_angle % angle_unit

            # 在 1-2 之间的一个像素梯度方向, 其在 1, 2 上的分量分别为: 1-(15°/45°), 15°/45°
            # 考虑当前像素的梯度大小, 则在 bin_size 个梯度方向上的增量为:
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    return orientation_centers


def get_image_cell_gradient(image, cell_size, bin_size):
    """
    将图像按 cell 分块, 求取每个 cell 的梯度直方图.
    :param image: 给灰度图像, 计算其每个像素点的梯度大小与方向.
    :param cell_size: 将图像分成大小为 cell_size 的单元,
    :param bin_size: 将 360 度方向分成多少种.
    :return: ndarray, shape=(int(h / cell_size), int(w / cell_size), bin_size)
    """
    h, w = image.shape[:2]

    gradient_magnitude, gradient_angle = get_gradient(image)
    # 将整张图像分成 height / cell_size × width / cell_size 块, 这里将多余的部分舍弃.
    cell_gradient_vector = np.zeros(shape=(int(h / cell_size), int(w / cell_size), bin_size))
    # print(cell_gradient_vector.shape)

    # 对按 cell 分块后的图像中的每个 cell 求取每个 cell 的梯度直方图.
    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            # 获取 cell 中每个像素的梯度.
            cell_magnitude = gradient_magnitude[i * cell_size: (i + 1) * cell_size,
                             j * cell_size: (j + 1) * cell_size]
            # 获取 cell 中每个像素的梯度方向.
            cell_angle = gradient_angle[i * cell_size: (i + 1) * cell_size,
                         j * cell_size: (j + 1) * cell_size]

            # print(cell_angle.max())
            # 求取每个 cell 的梯度直方图, 并赋值到当前图像的 cell 分块图中.
            cell_gradient_vector[i][j] = get_cell_gradient(cell_magnitude, cell_angle, bin_size)
    return cell_gradient_vector


def get_hog_image(image, cell_size, bin_size):
    """
    将 cell_gradient 图像转化成可视图片, 以展示效果.
    :param image: 原图像, 用于计算 cell_gradient 图.
    :param cell_size:
    :param bin_size:
    :return:
    """
    angle_unit = 360 / bin_size
    # 与原图大小相同的 hog_image
    hog_image = np.zeros(image.shape)
    cell_gradient = get_image_cell_gradient(image, cell_size=cell_size, bin_size=cell_size)
    # 数值归一化.
    cell_gradient = cell_gradient / np.array(cell_gradient).max()

    for i in range(cell_gradient.shape[0]):
        for j in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[i][j]
            angle = 0
            angle_gap = angle_unit
            for magnitude in cell_grad:
                # 角度转弧度
                angle_radian = math.radians(angle)
                # 在图像中每个 cell 的范围内画线, 以表达该 cell 梯度的方向与大小.
                # 求出 cell 的中心点坐标.
                x = int(i * cell_size + cell_size / 2)
                y = int(j * cell_size + cell_size / 2)
                # 图像中, 该线的长度应在 cell 范围内. 大小为: lengh = magnitude * cell_size
                lengh = magnitude * cell_size
                # 根据该梯度值所属的方向. 求取 x1, y1, x2, y2
                x1 = int(x + (lengh / 2) * math.cos(angle_radian))
                y1 = int(y + (lengh / 2) * math.sin(angle_radian))
                # x2 = int(x)
                # y2 = int(y)
                x2 = int(x - (lengh / 2) * math.cos(angle_radian))
                y2 = int(y - (lengh / 2) * math.sin(angle_radian))
                # 画线, 线的颜色深浅由梯度值的大小决定.
                cv.line(hog_image, (y1, x1), (y2, x2), color=int(255 * math.sqrt(magnitude)))
                # 下一个梯度值的角度方向为 angle:
                angle += angle_gap
    result = np.array(hog_image, dtype=np.uint8)
    return result


def get_hog_vector(image):
    # 求取图像在 8*8 的 cell 粒度下, 分 8 个梯度方向的梯度直方图矩阵.
    cell_gradient_vector = get_image_cell_gradient(image, cell_size=8, bin_size=8)

    hog_vector = []
    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):
            # 将每 4 个 cell 作为一个 block.
            block_vector = []
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            block_vector = np.array(block_vector)

            # 求 block_vector 向量的平方和再开方.
            magnitude = np.sqrt(np.sum(np.square(block_vector)))
            # 对 block_vector 中的梯度值除以其总和, 得出其各自所占的比例.
            if magnitude != 0:
                block_vector = block_vector / magnitude

            hog_vector.append(block_vector)
    # print(np.array(hog_vector).shape)
    return np.array(hog_vector)


def demo1():
    image = cv.imread('../dataset/data/other/silver.jpg', cv.IMREAD_GRAYSCALE)

    # # 显示图像 HOG 图.
    hog_image = get_hog_image(image, cell_size=8, bin_size=8)
    show_image(hog_image)
    return


def demo2():
    image = cv.imread('../dataset/data/other/silver.jpg', cv.IMREAD_GRAYSCALE)

    # 显示图像 HOG 图.
    result = get_hog_vector(image)
    print(result.shape)
    return


if __name__ == '__main__':
    demo1()
