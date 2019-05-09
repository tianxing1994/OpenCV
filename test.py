
import cv2 as cv
import numpy as np


def watershed_demo(image):
    print(image.shape)
    # 对图像进行模糊, 灰度化, 二值化操作.
    # 由于硬币之间是挨着的. 所以二值化图像中圆与圆之间有相连的区域, 这使得我们不能得到每个硬币的边缘.
    # 因此, 这里需要使用分水岭算法.
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow("binary-image", binary)

    # morphology operation, 开操作, 去除噪点.
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)

    # 膨胀操作. 膨胀后的图片 sure_bg 黑色区域可以被认为是确定一定是图片背景的部分.
    # sure_bg: sure background.
    sure_bg = cv.dilate(mb, kernel, iterations=3)
    # cv.imshow('mor-opt', sure_bg)

    # 使用开操作之后的 mb 图像进行距离转换计算.
    dist = cv.distanceTransform(mb, cv.DIST_L2, 3)
    # dist 中的值为浮点数, 被当作 0-1 之间的灰度图. 由于大部分都大于 1, 此时显示图像. 其为白色.
    # cv.imshow("distance-t", dist)
    # 将 dist 缩放到 0-1 之间.
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    # 将 dist 转化为 dist_output 后, 乘以 50 时, 其值都为正数, 被当作 0-255 之间的灰度图.
    # cv.imshow("distance-t", dist_output * 50)

    # 将 dist 中值较小, 也应是靠近硬币边缘的像素变成 0, 其它的变为 255.
    # 则白色面积变小, 我们可以认为, 此图中的白色区域确定是属于硬币的区域.
    # sure_fg: sure foreground
    ret, sure_fg = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    # cv.imshow("surface-bin", sure_fg)

    # 则,: sure_background - sure_foreground 的部分就是 unknown 不确定的, 边界在其中的区域.
    unknown = cv.subtract(sure_bg, sure_fg)

    # 生成标记图像 markers, sure_bg 中为 255 的部分被标记为 1, 为 0 的部分被标记为 0.
    # 这里被标记为 0 的部分是确定为 background 背景的部分.
    ret, markers = cv.connectedComponents(sure_bg)

    # watershed transform
    # 将 markers 标记数组 +1. 则: 确定为 background 背景的部分被标记为 1, 其它的被标记为 2.
    markers = markers + 1
    # 在 markers 中, 将 unknown 中为 255 的部分, 即边界在其中的部分改为 0.
    # 即: 标记, 存在边界的不确定区域为 0, 确定为 background 背景的区域为 1, 确定为硬币的区域为 2.
    markers[unknown==255] = 0

    # 使用此 markers 对原图像进行分水岭操作. 并在 markers 中, 将检测到的边界处标记为 -1.
    markers = cv.watershed(image, markers=markers)

    # 将边界像素改为红色. 以在图像中显示 watershed 算法得到的边界.
    image[markers==-1] = (0, 0, 255)
    cv.imshow("result", image)
    return


if __name__ == '__main__':
    src = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/coins.jpg")
    # cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    # cv.imshow("input image", src)

    watershed_demo(src)

    cv.waitKey(0)
    cv.destroyAllWindows()