"""
相关函数:
cv2.flip
cv2.VideoCapture
"""
import cv2 as cv
import numpy as np


def video_demo():
    """
    调用电脑的摄像头
    :return:
    """
    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        cv.imshow("video", frame)
        c = cv.waitKey(50)
        if c == 27:
            break


def get_image_info(image):
    """
    打印 imread() 读取到的图片信息.
    :param image:
    :return:
    """
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)
    print(np.array(image))


if __name__ == '__main__':
    image_path = 'C:/Users/tianx/PycharmProjects/opencv/dataset/lena.png'
    src = cv.imread(image_path)
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)
    get_image_info(src)
    # video_demo()
    # 保存图片
    cv.imwrite('C:/Users/tianx/PycharmProjects/opencv/dataset/lena_imwrite.png', src)
    cv.waitKey(0)
    cv.destroyAllWindows()