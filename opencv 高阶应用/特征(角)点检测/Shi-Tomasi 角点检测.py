"""
参考链接:
https://www.cnblogs.com/tianyalu/p/5468095.html
https://github.com/makelove/OpenCV-Python-Tutorial

Shi-Tomasi 角点检测 (适合于跟踪的图像特征)
Shi-Tomasi 算法是 Harris 算法的改进. Harris 算法最原始的定义是将矩阵 M 的行列式值与 M 的迹相减,
再将差值同预先给定的阈值进行比较. 后来 Shi-Tomasi 提出改进的方法, 若两个特征值中较小的一个大于最小阈值, 则会得到强角点.

算法步骤:
1. 计算灰度图像 I, 形状为 (m, n) 中每个像素在 x, y 方向上的一阶导数,
得到 Ixy, 形状为 (m, n, 2) 其中两个通道分别是每个像素在 x, y 方向上的梯度.

2. 对于每一个像素, 在给定的 ksize 窗口内, 统计每一个像素在 x, y 方向上的梯度 Ix, Iy, 将每一组 (Ix, Iy) 作为一组样本.
得到数组 M_, 形状为 (ksize*ksize, 2),

3. 计算 M_ 的协方差矩阵, 并求出协方差矩阵的 λ1, λ2 特征值
 (应该理解, 协方差矩阵的特征向量其实表征的是图像上梯度的主方向和次方向: 一个曲面, 曲率最大的方向, 与其垂直方向).

4. Shi-Tomasi 算法认为, 当  λ1, λ2 中的较小值大于给定阈值时, 该点为角点.
R = min(λ1, λ2)
在 Harris 角点检测中使用的是:
R = λ1*λ2 - k*(λ1 + λ2)^2
其中 k 是经验值, 一般取 0.04-0.06 之间

5. 最后, 非极大值抑制. 将所有得到的角点按角点较小 λ 的值大小进行降序排序, 也就是 R = min(λ1, λ2) 越大, 则是强角点.
从第一个角点开始, 将距离该角点距离小于阈值的删除, 得到过滤后的角点.
"""
import numpy as np
import cv2 as cv


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


image_path = '../../dataset/data/image_sample/image00.jpg'
image = cv.imread(image_path)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray, maxCorners=25, qualityLevel=0.01, minDistance=10)

print(corners.shape)
# corners 形如:  [[[173, 138]], [[141, 63]], [[156, 108]], [[81, 17]], [[115, 11]]], 形状为 (25, 1, 2)
corners = np.array(corners, dtype=np.int)
for i in corners:
    x, y = i.ravel()
    cv.circle(img=image, center=(x, y), radius=2, color=(0, 0, 255), thickness=-1)

show_image(image)
