"""
https://www.cnblogs.com/denny402/p/5167414.html
"""
from skimage import morphology, draw, color, data
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def skeletonize_demo():
    image = np.zeros((400, 400))

    image[10:-10, 10:100] = 1
    image[-100:-10, -10:-10] = 1
    image[10:-10, -100:-10] = 1

    rs, cs = draw.line(250, 150, 10, 280)

    for i in range(10):
        image[rs + i, cs] = 1
    rs, cs = draw.line(10, 150, 250, 280)
    for i in range(20):
        image[rs + i, cs] = 1

    ir, ic = np.indices(image.shape)

    # (x-a)^2 + (y-b)^2 < c^2
    circle1 = (ic - 135) ** 2 + (ir - 150) ** 2 < 30 ** 2
    circle2 = (ic - 135) ** 2 + (ir - 150) ** 2 < 20 ** 2
    image[circle1] = 1
    image[circle2] = 0

    skeleton = morphology.skeletonize(image)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('original', fontsize=20)

    ax2.imshow(skeleton, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('skeleton', fontsize=20)

    fig.tight_layout()
    plt.show()
    return


def skeletonize_demo2():

    # image_path = r'C:\Users\Administrator\PycharmProjects\openCV\dataset\data\shape_sample\18.png'
    # image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    # image = np.uint8(image / 255)

    image = color.rgb2gray(data.horse())
    image = 1-image

    skeleton = morphology.skeletonize(image)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('original', fontsize=20)

    ax2.imshow(skeleton, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('skeleton', fontsize=20)

    fig.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    skeletonize_demo2()