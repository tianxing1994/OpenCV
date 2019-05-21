"""
相关函数:
cv2.adaptiveThreshold
cv2.threshold
"""
import cv2 as cv
import numpy as np


def big_image_binary1(image):
    print(image.shape)
    cw = 256
    ch = 256
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:col+cw]
            # ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 127, 20)
            gray[row:row+ch, col:col+cw] = dst
            print(np.std(dst), np.mean(dst))
    cv.imwrite("C:/Users/tianx/PycharmProjects/opencv/dataset/image1.jpg", gray)
    return


def big_image_binary2(image):
    print(image.shape)
    cw = 256
    ch = 256
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:col+cw]
            print(np.std(roi), np.mean(roi))
            dev = np.std(roi)
            if dev < 15:
                gray[row:row+ch, col:col+cw] = 255
            else:
                ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
                gray[row:row+ch, col:col+cw] = dst
    cv.imwrite("C:/Users/tianx/PycharmProjects/opencv/dataset/image1.jpg", gray)
    return


if __name__ == '__main__':
    src = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/image0.JPG")
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)

    big_image_binary2(src)

    cv.waitKey(0)
    cv.destroyAllWindows()