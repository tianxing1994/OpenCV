import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


image_path = 'dataset/local_dataset/snapshot_1572484867.jpg'
image = cv.imread(image_path)
image_lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
image_l = image_lab[:, :, 0]
image_l = cv.convertScaleAbs(image_l)
# _, binary = cv.threshold(image_l, 127, 255, cv.THRESH_BINARY)
_, binary = cv.threshold(image_l, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
show_image(binary)


