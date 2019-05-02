import cv2 as cv
import numpy as np


def access_piexls(image):
    """
    遍历图像中的每一个像素, 把它相对于 255 取反
    :param image:
    :return:
    """
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("width: %s, height: %s channels: %s" % (width, height, channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv
    return image


def inverse(image):
    """
    调用 cv2 中的 API 接口, 将图片中的像素值取反
    :param image:
    :return:
    """
    dst = cv.bitwise_not(image)
    return dst


def create_image():
    img = np.zeros([400, 400, 3], np.uint8)
    img[:, :, 0] = np.ones([400, 400]) * 255
    cv.imshow("new_image", img)


if __name__ == '__main__':
    image_path = 'C:/Users/tianx/PycharmProjects/opencv/dataset/lena.png'
    src = cv.imread(image_path)
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)

    t1 = cv.getTickCount()
    # create_image()
    # 调用接口, 其底层采用 C 实现, 函数用时约 0.6 ms
    src = inverse(src)
    # 使用 python 实现, 函数用于约 1.7 s
    # src = access_piexls(src)
    t2 = cv.getTickCount()
    print("time: %s ms" % ((t2-t1)/cv.getTickFrequency() * 1000))

    cv.imshow("input image", src)

    cv.waitKey(0)
    cv.destroyAllWindows()