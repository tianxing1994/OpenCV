import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    image_path = 'C:/Users/tianx/PycharmProjects/opencv/dataset/example.png'
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    hist = cv.calcHist([image], [0, 1], None, [32, 32], [0, 180, 0, 256])
    plt.imshow(hist, cmap='gray')
    plt.title('HSV')
    plt.show()