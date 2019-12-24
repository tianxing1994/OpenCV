import cv2 as cv
import numpy as np


image = np.zeros(shape=(20, 20))
image[4:16, 4: 16] = 1
# print(image)

# 根据图像中的点在目标图像中的位置, 将原图转换为目标图像.
src = np.array([[4, 4], [4, 16], [16, 4], [16, 16]], dtype=np.float32)
dst = np.array([[8, 2], [12, 3], [3, 19], [17, 18]], dtype=np.float32)
perspective_matrix = cv.getPerspectiveTransform(src=src, dst=dst)
result = cv.warpPerspective(src=image, M=perspective_matrix, dsize=(20, 20), flags=cv.INTER_NEAREST)
print(result)

# 指定图像的四个角点与目标图像中的内容部分的四个点获取透视变换, 将目标图像中的内容部分校正.
src = np.array([[0, 0], [0, 20], [20, 0], [20, 20]], dtype=np.float32)
dst = np.array([[8, 2], [12, 3], [3, 19], [17, 18]], dtype=np.float32)
perspective_matrix = cv.getPerspectiveTransform(src=src, dst=dst)
result = cv.warpPerspective(src=result, M=perspective_matrix, dsize=(20, 20), flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP)
print(result)