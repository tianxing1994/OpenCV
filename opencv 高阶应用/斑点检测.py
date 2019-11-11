"""
参考链接:
https://blog.csdn.net/zhaocj/article/details/44886475
https://blog.csdn.net/qq_30815237/article/details/87282468
https://www.learnopencv.com/blob-detection-using-opencv-python-c/
https://github.com/makelove/OpenCV-Python-Tutorial

### 什么是斑点
图像特征点检测包括角点和斑点, 斑点是指二维图像中和周围颜色有颜色差异和灰度差异的区域,
因为斑点代表的是一个区域, 所以其相对于单纯的角点, 具有量多好的稳定性和更好的抗干扰能力.
斑点通常是指与周围有着颜色和灰度差别的区域. 在实际地较中, 往往存在着大量这样的斑点,
如一颗树是一个斑点, 一块草地是一个斑点, 一栋房子也可以是一个斑点. 由于斑点代表的是一个区域,
相比单纯的角点, 它的稳定性好, 抗噪声能力强, 所以它在图像配准上扮演了很重要的角色.
同时有时图像中的斑点也是我们关心的区域, 比如在医学与生物领域,
我们需要从一些 X 照片或细胞显微照片中提取一些具有特殊意义的斑点的位置或数量.

视觉领域的斑点检测的主要思路是检测出图像中比周围像素灰度大或比周围区域灰度值小的区域, 一般来说,
有两种基本方法:
1. 基于求导的微分方法, 这成为微分检测器
2. 基于局部极值的分水岭算法, OPENCV 中提供了 simpleBlobDetector 特征检测器来实现这种基本的斑点检测算法.

### LOG 斑点检测
使用高斯拉普拉斯算子检测图像斑点是一种比较常见的办法, 对一个二维的高斯函数 G(x, y, σ)
G(x, y; σ)  = 1 / 2πσ^2 * e^(- (x^2+y^2) / 2σ^2)

对高斯函数在 x, y 方向上分别求取二阶导数后相加, 得到拉普拉斯算子. 此算子与图像进行卷积, 可检测斑点.
将一个图像与一个卷积核进行卷积, 可以看作是比较图像部分与卷积核的相似度.

### 基于局部极值的分水岭算法, 斑点检测 simpleBlobDetector
这种检测方法步骤:
1. 对一张图片设定一个低阈值 minThreshold, 一个高阈值 maxThreshhold, 及步进间隔 thresholdStep.
对 [minThreshold, maxThreshold] 区间, 使用 thresholdStep 作为步进, 对图像进行二值化. 得到一系列图像.
2. 对每一张二值图片, 使用 findContours 查找这些图像的轮廓, 并计算每一个轮廓的中心.
3. 将这一系列的, 所有的轮廓放到一起. 对比这些轮廓的中心, 将中心距离小于 theminDistBetweenBlobs 值的归为一个分组.
每一个分组则代表了一个 blob 斑点特征.
4. 每一个分组代表一个斑点,
斑点的位置: 为所有组内轮廓的中心坐标的加重平均值, 权值等于该轮廓的惯性率平方, 即轮廓的圆度.
斑点的大小: 则是组内轮廓面积大小居中的半径大小.
5. 以上得出的所有斑点, 实际上是图像中所有独立的块. 并不是所有这些都是我们心目中的斑点, 我们通过一些限定条件来得到更准确的斑点.
如: 颜色 (blobColor), 面积 (minArea), 形状.
形状则可以用圆度 (minCircularity), 偏心率 (minInertiaRatio), 凸度 (minConvexity) 来表示.
对于二值图像来说, 只有两种斑点颜色, 白色斑点和黑色斑点, 我们只需要一种颜色的斑点, 通过确定斑点的灰度值就可以区分斑点的颜色.

圆形的斑点是最理想的, 任意形状的圆度 C 定义为:
C = 4πS / p^2
其中, S 和 p 分别表示该形状的面积和周长.
当 C 为 1 时, 表示该形状是一个完美的圆形, 而当 C 为 0 时, 表示该形状是一个逐渐拉长的多边形.

偏心率是指某一个椭圆轨道与理想圆形的偏离程度, 长椭圆轨道的偏心率高, 而近于圆形的轨道的偏心率低.
圆形的偏心率等于 0. 椭圆的偏心率介于 0 和 1 之间. 而偏心率等于 1 表示的是抛物线. 直接计算斑点的偏心率较为复杂,
但利用图像矩的概念计算图形的惯性率, 再由惯性率计算偏心率较为方便, 偏心率 E 和惯性率 I 之间的关系为:
E^2 + I^2 = 1
圆形的偏心率平方加惯性率平方等于 1, 因此, 惯性率越接近 1, 圆形的程度越高.

最后一个表示斑点形状的量是凸度. 在平面中, 凸形图指的是图形的所有部分都在由该图形切线所围成的区域的内部.
我们可以用凸度来表示斑点凹凸的程度, 凸度 V 的定义为:
V = S / H
其中, H 表示该斑点的凸壳的面积.
在计算斑点的面积, 中心坐标, 尤其是惯性率时, 都可以应用图像矩的方法.

### 矩
矩在统计学中被用来反映随机变量的分布情况, 推广到力学中, 它被用来描述空间物体的质量分布.
同样的道理, 如果我们将图像的灰度值看作是一个二维的密度分布函数, 那么矩方法即可用于图像处理领域.
设 f(x, y) 是一幅数字图像, 则它的矩 Mij 为:
M_{ij} = ∑∑x^{i}y^{j}f(x, y)

对于二值图像来说, 零阶矩 M_{00} 等于它的面积, 图形的质心为:
{x', y'} = {M_{10} / M_{00}, M_{01} / M_{00}}.

图像的中心矩 μ_{pq} 定义为:
μ_{pq} = ∑∑(x-x')^{p}(y-y')^qf(x, y)

一阶中心矩称为静矩, 二阶中心矩称为惯性矩. 如果仅考虑二阶中心矩的话, 则图像完全等同于一个具有确定的大小,
方向和离心率, 以图像质心为中心具具有恒定辐射度的椭圆. 图像的协方差矩阵为:
cov[f(x, y)] = [[μ'_{20}, μ'_{11}], [μ'_{11}, μ'_{02}]] =
[[μ_{20} / μ_{00}, μ_{11} / μ_{00}], [μ_{11} / μ_{00}, μ_{02} / μ_{00}]]

该矩阵的两个特征值 λ_{1} λ_{2} 对应于图像强度(椭圆) 的主轴和次轴.
参考链接: https://blog.csdn.net/zhaocj/article/details/44886475
"""

import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    # cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    # image_path = '../dataset/data/other/blob.jpg'
    image_path = '../dataset/data/other/silver.jpg'
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    detector = cv.SimpleBlobDetector_create()
    keypoints = detector.detect(image)

    # flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 参数, 确保圆的大小对应于斑点的大小.
    result = cv.drawKeypoints(image=image,
                              keypoints=keypoints,
                              outImage=np.array([]),
                              color=(0, 0, 255),
                              flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    show_image(result)
    return


def demo2():
    image = cv.imread("../dataset/data/other/blob.jpg", cv.IMREAD_GRAYSCALE)

    params = cv.SimpleBlobDetector_Params()

    # 阈值.
    params.minThreshold = 10
    params.maxThreshold = 200
    params.thresholdStep = 10

    params.minRepeatability = 2
    params.minDistBetweenBlobs = 10

    # 按面积过滤, 指定可接受的最小面积
    params.filterByArea = True
    params.minArea = 1500

    # 按圆度过滤, 指定圆度最小值.
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # 按凸性过滤,
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # 按惯量过滤,
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # 默认检测默色斑点, 如果需要检测白色斑点, 设置 filterByColor 为 True 并将 blobColor 设置为 255
    # params.filterByColor = True
    # params.blobColor = 255

    detector = cv.SimpleBlobDetector_create(parameters=params)
    keypoints = detector.detect(image)

    # flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 参数, 确保圆的大小对应于斑点的大小.
    result = cv.drawKeypoints(image=image,
                              keypoints=keypoints,
                              outImage=np.array([]),
                              color=(0, 0, 255),
                              flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    show_image(result)
    return


def demo3():
    image_path = '../dataset/data/other/blob.jpg'
    # image_path = '../dataset/data/other/silver.jpg'
    # image_path = '../dataset/data/other/silverhome.jpg'
    # image_path = '../dataset/local_dataset/snapshot_1572510101.jpg'
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    # image = cv.resize(image, dsize=None, fx=0.2, fy=0.2, interpolation=cv.INTER_LINEAR)

    params = cv.SimpleBlobDetector_Params()

    # 阈值.
    params.minThreshold = 10
    params.maxThreshold = 240
    params.thresholdStep = 5

    params.minRepeatability = 2
    params.minDistBetweenBlobs = 10

    # 按面积过滤, 指定可接受的最小面积
    params.filterByArea = True
    params.minArea = 200

    # 按圆度过滤, 指定圆度最小值.
    params.filterByCircularity = True
    params.minCircularity = 0.01

    # 按凸性过滤,
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # 按惯量过滤,
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # params.filterByColor = True
    # params.blobColor = 255

    detector = cv.SimpleBlobDetector_create(parameters=params)
    keypoints = detector.detect(image)

    # flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 参数, 确保圆的大小对应于斑点的大小.
    result = cv.drawKeypoints(image=image,
                              keypoints=keypoints,
                              outImage=np.array([]),
                              color=(0, 0, 255),
                              flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    show_image(result)
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()
