"""
cv2.getPerspectiveTransform
根据 4 对对应点计算透视变换.

语法:
Python: cv.GetPerspectiveTransform(src, dst, mapMatrix) → None
参数:
src: 源图像中四边形顶点的坐标.
dst: 目标图像中相应四边形顶点的坐标.

该函数计算 3*3 的透视变换矩阵 M, 以便执行透视变换操作. (将 M 与 src 源图矩阵相乘, 得到 dst 目标图像)


cv2.warpPerspective
将透视变换应用于图像.

语法:
Python: cv2.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst
参数:
src: 输入图像
dst: 输出图像, 大小由 dsize 指定, 类型与 src 相同.
M: 3*3 的转换矩阵
dsize: 指定输出图像的大小.
flags: INTER_LINEAR 或 INTER_NEAREST 插值方式与可选的逆变换标志 WARP_INVERSE_MAP 的组合.
borderMode: 像素外推方法 (BORDER_CONSTANT 或 BORDER_REPLICATE)
borderValue: 在边界不变的情况下使用的值, 默认情况下它等于 0.
"""
import cv2 as cv
import numpy as np


def demo1():
    src = np.array([[0, 0], [0, 100], [100, 0], [100, 100]], dtype=np.float32)
    dst = np.array([[0, 10], [10, 10], [200, 20], [180, 300]], dtype=np.float32)
    result = cv.getPerspectiveTransform(src=src, dst=dst)

    print(result)
    return


def demo2():
    image = np.zeros(shape=(20, 20))
    image[4:16, 4: 16] = 1
    # print(image)

    # 根据图像中的点在目标图像中的位置, 将原图转换为目标图像.
    src = np.array([[4, 4], [4, 16], [16, 4], [16, 16]], dtype=np.float32)
    dst = np.array([[8, 2], [12, 3], [3, 19], [17, 18]], dtype=np.float32)
    perspective_matrix = cv.getPerspectiveTransform(src=src, dst=dst)
    result = cv.warpPerspective(src=image, M=perspective_matrix, dsize=(20, 20), flags=cv.INTER_NEAREST)
    print(result)

    # 指定图像的四个角点与目标图像中的内容部分的四个点获取透视变换, 将目标图像中的内容部分校正.
    src = np.array([[0, 0], [0, 20], [20, 0], [20, 20]], dtype=np.float32)
    dst = np.array([[8, 2], [12, 3], [3, 19], [17, 18]], dtype=np.float32)
    perspective_matrix = cv.getPerspectiveTransform(src=src, dst=dst)
    result = cv.warpPerspective(src=result, M=perspective_matrix, dsize=(20, 20), flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP)
    print(result)
    return


if __name__ == '__main__':
    demo2()
