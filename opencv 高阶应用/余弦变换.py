import numpy as np
import cv2 as cv


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    image_path = '../dataset/data/image_sample/example.png'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    data = np.array(gray, dtype=np.float32)
    show_image(gray)

    dct_data = cv.dct(data)
    dct_data_ = np.array(dct_data / np.max(dct_data) * 255)
    show_image(dct_data_)

    idct_data = cv.idct(dct_data)
    result = np.array(idct_data, dtype=np.uint8)
    show_image(result)
    return


if __name__ == '__main__':
    demo1()
