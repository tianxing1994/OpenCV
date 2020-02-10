import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


image_path = "dataset/data/image_sample/lena.png"
image = cv.imread(image_path)
image_ = cv.Laplacian(src=image, ddepth=cv.CV_8U, ksize=1)
show_image(image_)
# kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
# image = cv.filter2D(image, cv.CV_8U, kernel)
# show_image(image)