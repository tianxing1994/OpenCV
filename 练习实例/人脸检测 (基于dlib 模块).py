"""
参考链接:
https://www.jb51.net/article/133472.htm
https://blog.csdn.net/qiumokucao/article/details/81610628


这里还有一个基于 face_recognition 包的人脸识别演示.
https://blog.csdn.net/weixin_42738495/article/details/90183384


windows 下安装 dlib
https://blog.csdn.net/qq_35044509/article/details/78882316
# 安装有点麻烦, 好像需要有 CMake, 等, 因为我的电脑上有 C++ 相关的软件.
# 所以我不清楚到底依赖哪些. 总之, 我的装好了.
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cmake
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple boost
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple dlib

模型下载链接:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""
import sys
import numpy as np
import cv2 as cv
import dlib


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def rect_to_bounding_box(rect):
    """
    :param rect: 是 dlib 脸部区域检测的输出
    :return:
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def shape_to_ndarray(shape, dtype=np.int):
    """
    :param shape: 是 dlib 脸部特征检测的输出，一个 shape 里包含了脸部特征的 68 个点位.
    :param dtype:
    :return:
    """
    coords = np.zeros(shape=(68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def resize(image, width=1200):
    """
    在人脸检测程序的最后, 我们显示检测的结果图片, 这里做 resize 是为了避免图片过大, 超出屏幕范围.
    :param image: 就是我们要检测的图片.
    :param width:
    :return:
    """
    r = width * 1.0 / image.shape[1]
    dsize = (width, int(image.shape[0] * r))
    resized = cv.resize(image, dsize, interpolation=cv.INTER_AREA)
    return resized


def demo():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../dataset/data/face_detection/shape_predictor_68_face_landmarks.dat")

    image_path = '../dataset/data/face_detection/faces/face_3.jpg'
    image = cv.imread(image_path)
    image = resize(image, width=1200)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 检测人脸部的区域, 如果有多个人脸则会有多个区域.
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        # 检测人脸区域中人脸的 68 个特征点位置.
        shape = predictor(gray, rect)
        shape = shape_to_ndarray(shape)
        (x, y, w, h) = rect_to_bounding_box(rect)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for x, y in shape:
            cv.circle(image, (x, y), 2, (0, 0, 255), -1)

    show_image(image)
    return


if __name__ == '__main__':
    demo()
