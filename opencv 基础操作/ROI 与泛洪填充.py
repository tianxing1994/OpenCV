"""
泛洪填充: 类似于油漆桶工具.
相关函数:
cv2.floodFill
"""
import cv2 as cv
import numpy as np


def part_to_gray(image):
    part = image[50:200, 300:500]
    gray = cv.cvtColor(part, cv.COLOR_BGR2GRAY)
    backface = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    image[50:200, 300:500] = backface
    return image


def fill_color_demo(image):
    copy_image = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)
    cv.floodFill(copy_image, mask, (30, 30), (0, 255, 255),
                 (100, 100, 100), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
    return copy_image


def fill_binary():
    image = np.zeros([400, 400, 3], np.uint8)
    image[100:300, 100:300, :] = 255
    cv.imshow("fill_binary", image)

    mask = np.ones([402, 402, 1], np.uint8)
    mask[101:301, 101:301] = 0
    cv.floodFill(image=image,
                 mask=mask,
                 seedPoint=(200, 200),
                 newVal=(255, 0, 0),
                 upDiff=cv.FLOODFILL_MASK_ONLY)
    cv.imshow("filled_binary", image)


if __name__ == '__main__':
    image_path = 'C:/Users/tianx/PycharmProjects/opencv/dataset/lena.png'
    src = cv.imread(image_path)

    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)

    # src = part_to_gray(src)
    # src = fill_color_demo(src)
    # cv.imshow("face", src)
    fill_binary()

    cv.waitKey(0)
    cv.destroyAllWindows()