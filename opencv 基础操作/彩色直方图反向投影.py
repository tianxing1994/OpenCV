import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def calc_bgr_hist(image, bins=16):
    h, w, c = image.shape
    bgr_hist = np.zeros([bins * bins * bins, 1], np.float32)
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b / 256 * bins) * bins * bins + np.int(g / 256 * bins) * bins + np.int(r / 256 * bins)
            bgr_hist[np.int(index), 0] = bgr_hist[np.int(index), 0] + 1
    return bgr_hist


def calc_bgr_back_project(image, bins=16):
    bgr_hist = calc_bgr_hist(image, bins=bins)
    back_project = np.zeros(image.shape[:2], np.int)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            b, g, r = image[row, col, 0], image[row, col, 1], image[row, col, 2]
            index = np.int(b / 256 * bins) * bins * bins + np.int(g / 256 * bins) * bins + np.int(r / 256 * bins)
            value = bgr_hist[np.int(index), 0]
            back_project[row, col] = value
    back_project = 255 * back_project / np.max(back_project)
    back_project = np.uint8(back_project)
    return back_project


def demo1():
    image_path = '../dataset/data/id_card/id_card_1.jpg'
    image = cv.imread(image_path)
    result = calc_bgr_back_project(image)
    _, binary = cv.threshold(result, 127, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    show_image(binary)
    return


if __name__ == '__main__':
    demo1()
