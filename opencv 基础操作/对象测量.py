"""
相关函数:
cv2.threshold
cv2.findContours
cv2.contourArea
cv2.boundingRect
cv2.moments
cv2.circle
cv2.rectangle
cv2.drawContours
cv2.approxPolyDP
"""
import cv2 as cv
import numpy as np


def measure_object(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value", ret)
    cv.imshow("binary image", binary)
    dst = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    contours, hireachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        x, y, w, h = cv.boundingRect(contour)
        mm = cv.moments(contour)
        type(mm)
        cx = mm['m10'] / mm['m00']
        cy = mm['m01'] / mm['m00']
        cv.circle(image, (np.int(cx), np.int(cy)), 3, (0, 255, 255), -1)
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        approxCurve = cv.approxPolyDP(contour, 4, True)
        print(approxCurve.shape)

        if approxCurve.shape[0] > 6:
            cv.drawContours(dst, contours, i, (0, 255, 0), 2)
        if approxCurve.shape[0] == 4:
            cv.drawContours(dst, contours, i, (0, 0, 255), 2)
        if approxCurve.shape[0] == 3:
            cv.drawContours(dst, contours, i, (0, 255, 0), 2)

    cv.imshow("measure-contours", image)
    return


if __name__ == '__main__':
    src = cv.imread("C:/Users/tianx/PycharmProjects/opencv/dataset/contours.png")
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)

    measure_object(src)

    cv.waitKey(0)
    cv.destroyAllWindows()