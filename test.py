import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def FT(src):
    lab = cv.cvtColor(src, cv.COLOR_BGR2LAB)
    gaussian_blur = cv.GaussianBlur(src, (5, 5), 0)

    mean_lab = np.mean(lab, axis=(0, 1))
    # print(mean_lab.shape)

    salient_map = (gaussian_blur - mean_lab) * (gaussian_blur - mean_lab)
    result = np.sum(salient_map, axis=2)
    print(result)
    print(result.shape)

    result = (result - np.amin(result)) / (np.amax(result) - np.amin(result))

    return result


image_path = 'dataset/local_dataset/snapshot_1572484867.jpg'
image = cv.imread(image_path)

x, y, w, h = cv.selectROI(windowName="selectROI", img=image, showCrosshair=False)
roi = image[y: y+h, x: x+w]
salient_image = FT(roi)
show_image(salient_image)

