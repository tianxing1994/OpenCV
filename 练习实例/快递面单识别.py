"""
参考链接:
https://blog.csdn.net/huangwumanyan/article/details/82526873
"""
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 75, 125, apertureSize=3)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # edges = cv.dilate(edges, kernel)
    show_image(edges)
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    show_image(image)


def line_detect_possible_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 75, 125, apertureSize=3)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    edges = cv.dilate(edges, kernel)
    show_image(edges)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=0)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    show_image(image)


if __name__ == '__main__':
    # image_path = '../dataset/data/exercise_image/express_paper.jpg'
    image_path = '../dataset/data/exercise_image/express_paper_2.jpg'
    # image_path = '../dataset/data/exercise_image/express_paper_3.jpg'
    image = cv.imread(image_path)
    print(image.shape)
    line_detection(image)
