import time
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


image_path = '../dataset/image/snapshot_1572515563.jpg'
image = cv.imread(image_path)
# show_image(image)

roi = image[868:942, 422:500]
show_image(roi)
# filename = f'../test/login_roi/roi_qq_{str(int(time.time()))}.jpg'
# cv.imwrite(filename, roi)
