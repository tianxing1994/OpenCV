# coding=utf8
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


win_size = (32, 32)
win_stride = (4, 4)
block_size = (8, 8)
block_stride = (4, 4)
cell_size = (4, 4)
nbins = 9


def get_roi_by_bounding_box(gray, bounding_box, angle):
    """
    将灰度图像先旋转再截取 ROI.
    :param gray:
    :param bounding_box:
    :param angle: 指将图像逆时针旋转的角度.
    :return:
    """
    x, y, w, h = bounding_box
    center_x = int(x + w // 2)
    center_y = int(y + h // 2)
    center = (center_x, center_y)

    h_, w_ = gray.shape
    m = cv.getRotationMatrix2D(center=center, angle=angle, scale=1)
    gray_rotated = cv.warpAffine(gray, M=m, dsize=(h_, w_))
    result = gray_rotated[y:y + h, x:x + w]
    return result


if __name__ == '__main__':
    bounding_box = np.array([79, 256, 35, 31])
    # roi 图像应保证至少有一个 hog window 的大小. 应先调整 roi 的大小.
    x, y, w, h = bounding_box
    w = w if w > win_size[0] else win_size[0]
    h = h if h > win_size[1] else win_size[1]
    bounding_box = np.array([x, y, w, h])

    image_path = "../dataset/image/luosi.jpg"
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    result = get_roi_by_bounding_box(gray=gray, bounding_box=bounding_box, angle=30)
    print(result.shape)
    show_image(result)
