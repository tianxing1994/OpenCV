import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    # cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


image_path = '../dataset/image/luosi.jpg'

image = cv.imread(image_path)

# (196, 558, 66, 69)
# Select a ROI and then press SPACE or ENTER button!
# Cancel the selection process by pressing c button!
cv.namedWindow("selectROI", cv.WINDOW_NORMAL)
bbox = cv.selectROI(windowName="selectROI", img=image, showCrosshair=False)
print(bbox)
# show_image(image)
