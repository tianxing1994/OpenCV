import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    src = np.array([[0, 0], [200, 0], [0, 200]], dtype=np.float32)
    dst = np.array([[0, 0], [100, 0], [0, 100]], dtype=np.float32)

    # 将源图像缩下 0.5 倍.
    affine_transform = cv.getAffineTransform(src, dst)
    print(affine_transform)
    return


def demo2():
    image = np.zeros(shape=(20, 20))
    image[4:16, 4: 16] = 1
    print(image)

    m = np.array([[0.5, 0, 0], [0, 0.5,  0]], dtype=np.float32)

    result = cv.warpAffine(src=image, M=m, dsize=(20, 20), borderValue=5)
    print(result)
    return


def demo3():
    image_path = '../dataset/data/image_sample/bird.jpg'
    image = cv.imread(image_path)
    show_image(image)
    h, w, _ = image.shape
    # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
    M = cv.getRotationMatrix2D(center=(h // 2, w // 2), angle=45, scale=0.5)
    # M = cv.getRotationMatrix2D(center=(0, 0), angle=45, scale=0.5)

    print(M)
    result = cv.warpAffine(image, M=M, dsize=(h, w))
    show_image(result)
    return


if __name__ == '__main__':
    demo3()
