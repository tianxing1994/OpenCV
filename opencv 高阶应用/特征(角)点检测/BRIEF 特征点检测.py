"""
### BRIEF 描述符
传递的特征点描述子如 SIFT, SURF 描述子, 每个特征点采用 128 维(SIFT) 或 64 维(SURF) 向量去描述, 每个维度上占用 4 个字节,
SIFT 需要 128*4=512 字节的内存, SURF 则需要 256 个字节. 如果对于内存资源有限的情况, 这种描述子方法显然不适应. 同时,
在形式描述子的过程中, 也比较耗时. 后来有人提出采用 PCA 降维的方法, 但没有解决计算描述子耗时的问题.

Binary Robust Independent Elementary Features. BRIEF 描述子采用二进制码串 (每一位非 1 即 0) 作为描述子向量,
论文中考虑长度有 128, 266, 512 几种 (OpenCV 里默认使用 256, 但是使用字节表示它们的, 所以这些值分别对应于 16, 32, 64),
同时, 形成描述子算法的过程简单, 由于采用二进制码串, 匹配上采用汉明距离, (一个串变成另一个串所需要的最小替换次数).
但由于 BRIEF 描述子不具有方向性, 大角度旋转会对匹配有很大的影响.

BRIEF 只提出了描述特征点的方法, 所以特征点的检测部分必须结合其他的方法, 如 SIFT, SURF 等, 但论文中建议与 FAST 结合,
因为会更能体现 BRIEF 速度快等优点.

CENSURE 特征检测器是一种尺度不变的中心环绕检测器 (CENSURE), 其性能优于其他检测器, 并且能够实时实现
CenSurE 在 OpenCV 中的实现是 STAR (cv2.xfeatures2d.StarDetector_create)

BRIEF 描述子原理简要为三个步骤, 长度为 N 的二进制码串作为描述子 (占用内存 N/8).
1. 以特征点 P 为中心, 取一个 S*S (48*48) 大小的 Patch 邻域.
2. 在这个邻域内随机取 N 对点, 然后对这 2*N 个点分别做高斯平滑, 比较 N 对像素点的灰度值的大小.
τ(p; x, y) := 1 if p(x) < p(y) else 0

其中, p(x), p(y) 分别是随机点 x=(u1, v2), y=(u2, v2) 的像素值.

3. 最后把步骤 2 得到的 N 个二进制码串组成一个 N 维向量即可:
f_{n_{d}}(P) := \sum{1<+i<=n_{d}}{s^{i-1}τ(p; x_{i}, y_{i})}



1. 测试前, 需要对随机点做高斯平滑, 由于采用单个的像素灰度值做经比较, 会对噪声很敏感;
采用高斯平滑图像, 会降低噪声的影响, 使得描述子更加稳定. 论文中建议采用 9*9 的 kernel.
2. 论文中对随机取 N 对点采用了 5 种不同的方法做测试, 论文中建议采用 G II 的方法.
(1) x_{i}, y_{i} 都均匀分布在 [-S/2, S/2]
(2) x_{i}, y_{i} 都高斯分布在 [0, S^{2} / 25]

3. 特征匹配是对利用的汉明距离进行判决, 直接比较两二进制码串的距离,
距离定义为: 其中一个串变成另一个串所需要的最小操作. 因而比欧氏距离运算速度快.
如果取 N=128, 即每个特征点需要 128/8=16 个字节内存大小作为其描述子.
· 两个特征编码对应 bit 位上相同元素的个数小于 64 的, 一定不是配对的.
· 一幅图上特征点与另一幅图上特征编码对应 bit 位上相同元素的个数最多的特征点配成一对.
"""
import numpy as np
import cv2 as cv


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

    star = cv.xfeatures2d.StarDetector_create()

    keypoints = star.detect(gray, None)

    image = cv.drawKeypoints(image=image, keypoints=keypoints, outImage=None, color=(0, 0, 255))
    show_image(image)
    return


def demo2():
    image_path = '../../dataset/data/image_sample/image00.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    star = cv.xfeatures2d.StarDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes=16, use_orientation=False)

    keypoints = star.detect(gray, None)

    keypoints, descriptor = brief.compute(gray, keypoints)
    print(len(keypoints))
    print(descriptor.shape)
    print(descriptor)
    return


if __name__ == '__main__':
    demo2()
