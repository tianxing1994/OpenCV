"""
参考链接:
https://blog.csdn.net/foreverhot1019/article/details/78793816
"""
import numpy as np
import cv2 as cv


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def binary_image(gray):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (17, 1))
    tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    show_image(tophat)

    gray_x = cv.Sobel(tophat, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
    gray_x = np.absolute(gray_x)
    min_value, max_value = np.min(gray_x), np.max(gray_x)
    gray_x = 255 * (gray_x - min_value) / (max_value - min_value)
    gray_x = gray_x.astype('uint8')

    gray_x = cv.morphologyEx(gray_x, cv.MORPH_CLOSE, kernel)
    _, binary = cv.threshold(gray_x, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    show_image(binary)

    return binary


def find_contours(binary):
    _, contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = list()
    for i, c in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(c)
        ar = w / h

        if (ar > 1.5 and ar < 4.0):
            if (w > 55 and w < 80) and (h > 15 and h < 30):
                filtered_contours.append((x, y, w, h))
    filtered_contours = sorted(filtered_contours, key=lambda x: x[0])
    return filtered_contours


def show_rectangle(image, filtered_contours):
    for x, y, w, h in filtered_contours:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    show_image(image)
    return


def demo():
    image_path = '../dataset/data/bank_card/card_ 5.png'
    image = cv.imread(image_path)
    image = cv.resize(image, dsize=(450, 280))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    binary = binary_image(gray)
    filtered_contours = find_contours(binary)

    print(filtered_contours)
    show_rectangle(image, filtered_contours)
    return


if __name__ == '__main__':
    demo()
