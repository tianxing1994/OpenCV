"""
相关函数:
cv.CascadeClassifier
cv.rectangle
cv.VideoCapture
cv.flip
"""
import cv2 as cv
import numpy as np


def face_detect_demo():
    """
    人脸检测: 在图像中找到人脸的位置.
    cv.CascadeClassifier() 实例化参数下载地址:
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt_tree.xml
    :param image:
    :return:
    """
    image = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/lena.png")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("C:/Users/tianx/PycharmProjects/opencv/dataset/data/haarcascade_frontalface_alt_tree.xml")
    faces = face_detector.detectMultiScale(gray, 1.02, 5)
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow("result", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def video_face_detect_demo():
    """
    在视频中进行人脸检测: 在图像中找到人脸的位置.
    cv.CascadeClassifier() 实例化参数下载地址:
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt_tree.xml
    :param image:
    :return:
    """
    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        image = cv.flip(frame, 1)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        face_detector = cv.CascadeClassifier("C:/Users/tianx/PycharmProjects/opencv/dataset/data/haarcascade_frontalface_alt_tree.xml")
        faces = face_detector.detectMultiScale(gray, 1.02, 5)
        for x, y, w, h in faces:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv.imshow("result", image)
        cv.waitKey(0)
        c = cv.waitKey(10)
        if c == 27:   # ESC
            break
        cv.destroyAllWindows()
    return


if __name__ == '__main__':
    # video_face_detect_demo()
    face_detect_demo()