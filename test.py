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
image1 = cv.GaussianBlur(image, ksize=(9, 9), sigmaX=1.6)
image2 = cv.Laplacian(src=image1, ddepth=cv.CV_64F, ksize=1)
image2 = np.abs(image2)
print(np.max(image2))
image2 = np.array(image2 / np.max(image2) * 255, dtype=np.uint8)
show_image(image2)