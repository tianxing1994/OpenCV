"""
参考链接:
https://blog.csdn.net/qq_30815237/article/details/87282468
https://www.learnopencv.com/blob-detection-using-opencv-python-c/
https://github.com/makelove/OpenCV-Python-Tutorial/blob/master
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
    image = cv.imread("../dataset/data/other/blob.jpg", cv.IMREAD_GRAYSCALE)

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

    params.thresholdStep = 10

    # 阈值.
    params.minThreshold = 10
    params.maxThreshold = 200

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


if __name__ == '__main__':
    # demo1()
    demo2()
