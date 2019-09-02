"""
参考链接:
https://blog.csdn.net/tengfei461807914/article/details/80978947

各种光流法 C++ 实现:
https://blog.csdn.net/xiaoyufei117122/article/details/53693627

光流法解释:
https://www.wandouip.com/t5i151565/



光流法, 我的理解:
对于目标追踪, 图像中目标部分的像素值不会凭空出现/凭空消失.
第一张图中 (x, y) 点处的像素值 I1(x, y) 必定会出现在第二张图中该点附近: I2(x+△x, y+△y).
这里我们要求取 △x 和 △y.
假设视频中两张图的取图间隔时间非常短, 则我们可以跟据灰度图像在该点处的梯度来求取 △x 和 △y.
△x = ∂I1/∂x * u,
△y = ∂I1/∂y * v,
其中: u, v 是点 (x, y) 在 x 和 y 方向上移动的距离. (在数字图像中是移动的像素点数).
有: I1(x, y) = I2(x + ∂I1/∂x*u, y + ∂I1/∂y*v)
其中: ∂I/∂x, ∂I/∂y 分别是 x, y 点处的图像偏导数(梯度), 我们可以通过第一张图来求取.

已知图1 (x, y) 处的像素在图2 中移动到了 (x + ∂I/∂x*u, y + ∂I/∂y*v) 处, 那么图2 中 (x, y) 处的像素值等于多少呢 ?
I2(x, y) = I1(x, y) + ∂I1/∂x*u + ∂I1/∂y*v

这里我们已知图1 和图2, 即已知 I1(x, y), I2(x, y), ∂I1/∂x, ∂I1/∂y, 求 u, v.
一个方程不能求两个未知数. 所以我们又认为两图中与 (x, y) 相近位置的像素的运动方向与 (x, y) 相同.
即: 以将 (x, y) 为中心的 9 个像素点代入以上方程都应该得到相同的 u, v 值. 所以这里就有了 9 个方程求 u, v.



光流: 光流的概念是指在连续的两帧图像当中, 由于图像中的物体移动或者摄像头的移动而使得图像中的目标的运动叫做光流.
(即: 考虑在摄像头不会动的表总况下, 如果视频中有一个运动目标, 那么这个视频中相邻两帧中运动的目标就是光流)
光流是一个向量场, 表示了一个点从第一帧运动到第二帧的移动.

光流有很多应用场景: 运动恢复结构, 视频压缩, 视频防抖动等.

光流法的工作原理基于如下假设:
1. 连续的两帧图像之间, 目标的像素亮度不改变.
2. 相邻的像素之间有相似的运动.
考虑第一帧的像素 I(x, y, t), 表示在时间 t 时像素 I(x, y) 的值, 在经过时间 dt 后, 此像素在下一帧移动了 (dx, dy).
因为这些像素是相同的, 而且亮度不变, 我们可以表示成, I(x, y, t) = I(x+dx, y+dy, t+dt).
假设移动很小, 使用泰勒公式可以表示成:
I(x+△x, y+△y, t+△t) = I(x, y, t) + ∂I/∂x * △x + ∂I/∂y * △y + ∂I/∂t * △t + H.O.T

H.O.T 是高阶无穷小.

由于第一个似设和使用泰勒公式展开的式子可以得到:
∂I/∂x * △x + ∂I/∂y * △y + ∂I/∂t * △t = 0

改写成:
∂I/∂x * △x/△t + ∂I/∂y * △y/△t + ∂I/∂t * △t/△t = 0

设
∂I/∂x = fx
∂x/∂t = u
∂y/∂t = v
fx * u + fy * v + ft = 0

上面公式就光叫光流方程, 其中 fx 和 fy 分别是图像的梯度, ft 是图像沿着时间的梯度. 但是 u, v 是未知的,
我们没办法用一个方程解两个未知数, 那么就有了 lucas-kanade 这个方法来解决这个问题.


Lucas-Kanade 算法
使用第二条假设, 就是所有的相邻像素都有相同的移动. LK 算法使用了一个 3×3 的窗口大小, 所以, 在这个窗口当中有 8 个像素点满足公式:
fx*u + fy*v + ft = 0.
将点代入方程, 现在的问题就变成了使用 9 个点求解两个未知量.
解的个数大于未知数的个数, 这是个超定方程, 使用最小二乘的方法来求解最优值. 如下为计算得到的结果.
[u, v].T = [[∑fxi^2, ∑fxi*fyi], [∑fxi*fyi, ∑fyi^2]].-1 · [-∑fxi*fti, -∑fyi*fti]

图像中逆矩阵与 Harris 角点检测很像, 说明角点是适合用来做跟踪的.

想法很简单, 给出一引起点用来跟踪, 从而获得点的光流向量. 但是有另外一个问题需要解决, 目前讨论的运动都是小步长的运动,
如果有大幅度的运动出现, 本算法就会失效.

使用的解决办法是利用图像金字塔. 在金字塔的顶端的小尺寸图片当中, 大幅度的运动就变成了小幅度的运动. 所以使用 LK 算法,
可以得到尺度空间上的光流.


两张灰度图像, 如果, 某一位置 (x, y) 处的灰度值第二张比第一张要小. 那么第一张图中的那个像素值移动去哪里了呢.
我们先求取该点 (x, y) 分别在 x, y 方向上的偏导, 第二张图的像素减小了, 则 I(x, y) = I(x + ∂I/∂x*u, y + ∂I/∂y*v)


这里，我们可以创建一个简单的应用，用来追踪视频中的一些点, 为了探测这些点, 我们使用cv2.goodFeaturesToTrack()来实现.

首先选取第一帧, 在第一帧图像中检测Shi-Tomasi角点, 然后使用LK算法来迭代的跟踪这些特征点.
迭代的方式就是不断向 cv2.calcOpticalFlowPyrLK() 中传入上一帧图片, 其中的特征点以及当前帧的图片.
函数会返回当前帧的点, 这些点带有状态1或者0, 如果在当前帧找到了上一帧中的点, 那么这个点的状态就是1, 否则就是0.
"""

import numpy as np
import cv2 as cv


cap = cv.VideoCapture("C:/Users/Administrator/Desktop/vtest.avi")

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))

# 从视频中读取一帧图片转换为灰度图像做为第一帧初始化图.
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

mask = np.zeros_like(old_frame)

while True:
    # 从视频中读取一帧图片转换为灰度图像
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old), in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)

    img = cv.add(frame, mask)
    cv.imshow('frame', img)

    k = cv.waitKey(30)
    if k == 27:
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
cap.release()
