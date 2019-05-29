import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt


def show_image(image):
    cv.namedWindow('input image', cv.WINDOW_NORMAL)
    cv.imshow('input image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


# 设置递归限制
sys.setrecursionlimit(15000)

original_image = cv.imread('../dataset/data/fruits.jpg', cv.IMREAD_GRAYSCALE)
original_image_boundaries = cv.imread('../dataset/data/fruits.jpg', cv.IMREAD_GRAYSCALE)

height, width = original_image.shape
# print(height, width)  # 480 512

R = np.zeros((height, width), dtype=np.int32)

neighbors = []
label_mean = []
visited = []


def recursive_label(i, j, label, intensity):
    if (i<0 or j<0 or i>=height or j>=width) or\
        R[i, j] != 0 or\
        np.absolute(original_image[i, j] - intensity)>20:
        return
    R[i, j] = label
    recursive_label(i, j + 1, label, original_image[i, j])
    recursive_label(i, j - 1, label, original_image[i, j])
    recursive_label(i - 1, j, label, original_image[i, j])
    recursive_label(i + 1, j, label, original_image[i, j])
    return


def recursive_merge(i, label_value):
    if i in visited:
        return
    visited.append(i)
    for neight_pixel in neighbors[i-1]:
        if np.absolute(label_mean[label_value-1] - label_mean[neight_pixel-1]) < 20:
            replace_pixel = np.where(R==neight_pixel)
            for k in range(0, len(replace_pixel[0])):
                x = replace_pixel[0][k]
                y = replace_pixel[1][k]
                R[x][y] = label_value
            recursive_merge(neight_pixel, label_value)
    return


def main():
    label = 1
    for i in range(height):
        for j in range(width):
            if R[i, j] == 0:
                recursive_label(i, j, label, original_image[i, j])
                label = label + 1

    label -= 1

    for x in range(1, label+1):
        neighbors.append([])
        r_pixel = np.where(R==x)
        for y in range(0, len(r_pixel[0])):
            i = r_pixel[0][y]
            j = r_pixel[1][y]

            if i>0:
                if R[i-1, j] != R[i, j] and R[i-1, j] not in neighbors[x-1]:
                    neighbors[x-1].append(R[i-1, j])
            if i<height-1:
                if R[i+1, j] != R[i, j] and R[i+1, j] not in neighbors[x-1]:
                    neighbors[x-1].append(R[i+1, j])
            if j<width-1:
                if R[i, j+1] != R[i, j] and R[i, j+1] not in neighbors[x-1]:
                    neighbors[x-1].append(R[i, j+1])
            if j>0:
                if R[i, j-1] != R[i, j] and R[i, j-1] not in neighbors[x-1]:
                    neighbors[x-1].append(R[i, j-1])

    label_list = []
    for i in range(1, label+1):
        r_pixel = np.where(R==i)
        for j in range(0, len(r_pixel[0])):
            x = r_pixel[0][j]
            y = r_pixel[1][j]
            label_list.append(original_image[x, y])
        label_mean.append(sum(label_list) / len(label_list))
        del label_list[:]

    for i in range(1, label+1):
        if i not in visited:
            recursive_merge(i, i)

    for i in range(1, R[height-1, width+1]):
        r_pixel = np.where(R==i)
        for j in range(0, len(r_pixel[0])):
            x = r_pixel[0][j]
            y = r_pixel[1][j]
            if x > 0:
                if R[x-1, y] != R[x, y]:
                    original_image_boundaries[x, y] = 255
            if x < height:
                if R[x+1, y] != R[x, y]:
                    original_image_boundaries[x, y] = 255
            if y < width:
                if R[x, y+1] != R[x, y]:
                    original_image_boundaries[x, y] = 255
            if y > 0:
                if R[x, y-1] != R[x, y]:
                    original_image_boundaries[x, y] = 255
    return


if __name__ == '__main__':
    main()
    show_image(original_image)
    show_image(original_image_boundaries)
    show_image(R * 255)



