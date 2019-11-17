"""
参考链接:
https://blog.csdn.net/weixin_37697191/article/details/89090686
https://github.com/makelove/OpenCV-Python-Tutorial

SIFT, SURF 算算是有专利保护的, 如果你使用它们, 就可能要花钱, 但 ORB 不需要.

ORB 基本是 FAST 关键点检测和 BRIEF 关键点描述器的结合体, 并修改增编辑器了性能.
首先, 它使用 FAST 找到关键点, 然后排序, 找到其中的前 N 个点. 它也使用图像金字塔从而产生尺度不变性特征.
数据的方差大的一个好处是使得特征更容易分辨.

### 对于描述符
ORB 使用 BRIEF 描述符. 但我们已经看到, 这个 BRIEF 的表现在旋转方面表现不佳.
因此, ORB 所做的是根据关键点的方向来 "引导".
对于在位置 (x_{i}, y_{i}) 的 n 个二进制测试的任何特性集, 定义一个包含这些像素坐标的 (n, 2) 矩阵.
然后利用补丁的方向, 找到旋转矩阵关旋转 S, 以得到引导 (旋转) 版本 S.

### 旋转不变性的实现原理:
1. BRIEF 算法对关键点的描述是在关键点附近按一定规则选取 n 对像素点,
对比每对像素点的大小得出 1 或 0 的返回结果, 串连这些结果, 得到长度为 n 的二进制串.
这个二进制串如: 100101, 可以转选成一个十进制数字来表示.
2. 当图片发生旋转时, 按原来的固定规则选取的 n 个像素点对, 将与之前的不一样, 这使得 BRIEF 不具有旋转不变性.
3. 需要一个方向表征, ORB 算法优化了像素对的选取方法, 首先将像素的值看作质量,
求得关键点附近一个区域内的质心, 将中心到质心的方向作为主方向. 在选点之前先进行旋转, 再选取, 则实现了旋转不变性.
4. 因为我们已预先实到了所要选的点对的规则, 所以我们只需要将这些点对的坐标进行旋转变换,
即可得到旋转后的点对坐标, 而不需要将关键点附近的整个窗口的像素进行旋转, 这节省了计算量.

ORB 将角度进行离散化, 以增加 2/30 (12度), 并构造一个预先计算过的简短模式的查找表.
只要关键点的方向是一致的, 就会使用正确的点集来计算它的描述符.

BRIEF 有一个重要的属性, 即每个比特的特性都有很大的方差, 而平均值接近 0.5. 但是一旦它沿着键点方向移动,
它就会先去这个属性并变得更加分散. 高方差使得特征更有区别. 因为它对输入的响应不同.
另一个可取的特性是让测试不相关, 因为每个测试都将对结果有所贡献. 为了解决所有这些问题,
ORB 在所有可能的二进制测试中运行一个贪婪的搜索, 以找到那结即有高方差又接近 0.5 的, 同时又不相关的结果, 被称为 rBRIEF.

### 对于描述符匹配
在传统的 LSH 上改进的多探测 LSH 是被使用的.
有说, ORB 比 SURF 快得多, 而且比 SURF 还好. 对于金景拼接的低功率设备, ORB 是不错的选择.

如特征点 A, B 的描述子如下:
A: 10101011
B: 10101010
判断 A B 的相似度: 对 A, B 作二进制的异或操作, 即可得到 A 和 B 的相似度. 而异或操作可以借助硬件完成, 具有很高的效率.
"""
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    image_path = '../../dataset/data/image_sample/image00.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create()
    keypoints = orb.detect(gray, None)
    keypoints, descriptors = orb.compute(gray, keypoints)
    image = cv.drawKeypoints(image, keypoints, None, color=(0, 0, 225), flags=0)
    show_image(image)
    return


def demo2():
    image_path = '../../dataset/data/image_sample/image00.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    fast = cv.FastFeatureDetector_create(threshold=32, nonmaxSuppression=True, type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16)

    keypoints = fast.detect(gray, None)

    brief = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes=64)
    keypoints, descriptor = brief.compute(gray, keypoints)

    image2 = cv.drawKeypoints(image, keypoints, None, color=(0, 0, 255))
    show_image(image2)
    return


if __name__ == '__main__':
    demo1()
    demo2()
