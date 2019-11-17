"""
参考链接:
https://blog.csdn.net/tostq/article/details/49314017
https://github.com/makelove/OpenCV-Python-Tutorial

FAST 特征检测器效果很好. 但是从实时处理的角度来看, 这些算法不够快.
一个最好的例子就是 SLAM 同步定位与地图构建. 移动机器人的计算资源非常有限.

### FAST 算法原理
FAST 的方法主要是考虑像素点附近的圆形窗口上的 16 个像素,
如果有 n 个连续的点都比中心像素 p 的强度都大, 或都小的话,
则这样的中心点就是角点, 实际上比较强度时, 需要加上阈值 t.

# I_{x} 比 I_{p} 更暗:
S_{p→x} = d, if I_{p→x} <= I_{p} - t
# I_{x} 与 I_{p} 相似:
S_{p→x} = s, if I_{p} - t <= I_{p→x} <= I_{p} + t
# I_{x} 比 I_{p} 更亮:
S_{p→x} = b, if I_{p} + t <= I_{p→x}

一般情况下, n 取 12, 所以这个标准定义为 FAST-12, 而实际上当 n=9 时, 往往能取得较好的效果.

注:
1. 当圆半径为 3 时, 圆环上像素为 16 个, 取 n=9 个.
2. 当圆半径为 2 时, 圆环上像素为 8 个, 取 n=5 个.
3. 当圆半径为 2.5 时, 圆环上像素为 12 个, 取 n=7 个.


如果要提高检测速度的话, 只需要检测 4 个点就可以了:
首先比较第 1 和第 9 个像素, 如果两个点像素强度都在中心像素强度 t 变化范围内 (及都同中心点相似), 则说明这不是角点,
如果接下来检测第 5 和 13 点时, 发现上述四点中至少有三个点同中心点不相似, 则可以说明这是个角点.
之后为了提高精度, 我们还可以对上面运算后的剩下的候选角点进行全部的 16 点检测, 从而确定其是不是角点.

但是这个方法也有缺点:
1. 如果首先的四点检测里, 只有 2 个点同中心点不相似, 也并不能说明这不是角点.
2. 检测的效率严重依赖于检测点的顺序和角点附近的分布, 很难说明所选择比较的像素位置能最好地反应角点性能.
3. 前面的四点检测结果没能充分用到后面检测上来.
4. 并连在一起的特征点很可能检测到了相邻的位置.
所以, 为了解决上述问题, 以下介绍通过机器学习来改善 FAST 算法的速度和通用性.

### 通过机器学习的方法
我们可以解决上面提到的前三个问题, 而第四个问题可以通过非极大值抑制来解决.
机器学习方法采用的是决策树方法, 通过对像素点进行分类, 找到角点附近的 16 点位置中最能区分这个分类的位置,
而叶则用来指明是否是角点.
主要分为两个部分: (没看懂, 大概是通过信息熵的方式学习出信息量最大的判断顺序.)
1. 首先给定 n, 对所有的 16 像素圆环建立 FAST-n 检测, 然后从一组图像内提取大量的角点.
2. 对 15 个像素的每个位置 x[1, 16], 将其同中心像素比较, 获得三个状态, 如下:

# I_{x} 比 I_{p} 更暗:
S_{p→x} = d, if I_{p→x} <= I_{p} - t
# I_{x} 与 I_{p} 相似:
S_{p→x} = s, if I_{p} - t <= I_{p→x} <= I_{p} + t
# I_{x} 比 I_{p} 更亮:
S_{p→x} = b, if I_{p} + t <= I_{p→x}

选择一个位置 x, 将所有训练图像中的所有像素点 (记为集 P) 同其该位置点进行上述等式比较,
分别计算每点的状态 S_{p→x}, 由此一来, 对于每个位置 x, 我们都可以得到一个状态集,
而每个状态就是指一个像素 (集 P 内的每个像素) 同其附近 (圆环上) 该位置像素的状态.

3. 然后我们选择一个 x 位置的状态集, 可以将集 P 内的像素点根据其在状态集对应位置上状态,
分成三个部分 P_{d}, P_{s} 和 P_{b}, 其意思是指该子集内的像素, 其附近圆环 x 位置的像素同比较,
是更暗 (d 集), 相似 (s 集), 或更亮 (b 集).

4. 之后再定义一个布尔变量 K_{p} 来确定当前像素点 p 是否是角点. 对于任意一个点集 Q, 我们可以计算总共的熵值:
H(Q) = (c + c') log_{2}(c + c') - clog_{c}(c) - c'log_{2}(c')
其中:
c = |{i ∈ Q: K_{i} is true}| (number of corners)
c' = |{i ∈ Q: K_{i} is false}| (number of noncorners)
对于一个位置 x 来说, 它分类的信息量为:
H_{g} = H(P) - H(P_{d}) - H(P_{s}) - H(P_{d})

5. 接下来, 我们就是选择信息量最大的 x 位置, 同时在其下面的子集 P_{d}, P_{s} 和 P_{b} 内继续迭代选择信息量最大的 x 位置
(每个子集内选择 x 位置可以分别写成 x_{d}, x_{s}, x_{b}), 然后继续将子集再次分割, 如将 P_{d} into P_{b.d}, P_{b.s}, P_{b.b}.
每一次分割所选的 x 都是通将集分类成拥有最大信息量的位置.

6. 为了更好的优化, 我们强制让 x_{d}, x_{s}, x_{b} 相等, 这样的话, 我们选择第二个测试的位置点将会一样,
只需要两个位置点的比较, 我们就排除绝大多数的点, 如此以来将让速度大大提高.


### 非极大值抑制
因为 FAST 方法没有计算角响应度, 所以我们很难采用常规的方法来直接进行非极大值抑制, 对于一个给定的 n,
如果阈值 t 增加, 那么检测的角点数将会减少, 因此: 角点强度可以被定义为如果这个点可以被检测成角点时,
t 所能取到的最大值.

决策树能根据一个给定的 t, 非常高效地确定一个点是否是角点, 由此一来, 我们可以通过变化 t,
来确定找到该点由角点变为非角点时的 t 值, 而这个 t 值就是让该检测为角点的最大阈值,
这个查找方法可以用二分法解决. 或者, 我们也可以用一个迭代的方法.

找到了角响应度的衡量后, 我们就可以应用原来的非极大值抑制方法了, 最终得到我们想要的角点.
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

    fast = cv.FastFeatureDetector_create()

    keypoints = fast.detect(gray, None)
    image2 = cv.drawKeypoints(image, keypoints, None, color=(0, 0, 255))

    show_image(image2)

    print("Threshold: ", fast.getThreshold())
    print("nonmaxSuppression: ", fast.getNonmaxSuppression())
    print("neighborhood: ", fast.getType())
    print("Total Keypoints with nonmaxSuppression: ", len(keypoints))

    # 关闭非极大值抑制.
    fast.setNonmaxSuppression(0)
    keypoints = fast.detect(gray, None)
    print("Total Keypoints without nonmaxSuppression: ", len(keypoints))

    image3 = cv.drawKeypoints(image, keypoints, None, color=(0, 0, 255))
    show_image(image3)

    return


def demo2():
    image_path = '../../dataset/data/image_sample/image00.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 1. 当圆半径为 3 时, 圆环上像素为 16 个, 取 n=9 个. cv.FAST_FEATURE_DETECTOR_TYPE_9_16
    # 2. 当圆半径为 2 时, 圆环上像素为 8 个, 取 n=5 个. cv.FAST_FEATURE_DETECTOR_TYPE_5_8
    # 3. 当圆半径为 2.5 时, 圆环上像素为 12 个, 取 n=7 个. cv.FAST_FEATURE_DETECTOR_TYPE_7_12
    fast = cv.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True, type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16)

    keypoints = fast.detect(gray, None)
    image2 = cv.drawKeypoints(image, keypoints, None, color=(0, 0, 255))

    show_image(image2)
    return


if __name__ == '__main__':
    demo2()
