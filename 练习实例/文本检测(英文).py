# coding=UTF-8
"""
参考链接:
https://blog.csdn.net/jkjj2015/article/details/87621668


预训练好的模型下载链接:
百度网盘: https://pan.baidu.com/s/1SGB-Sy4vBgwCo6yOJpqfQQ
提取码: 479j
"""
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2


def show_image(image, win_name='input image'):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


# 神经网络模型需要规定大小的输入图像.
# EAST文本要求输入图像尺寸为32的倍数. (width, height) -> (320, 320)
input_size = (320, 320)

# image_path = "../dataset/data/airport/airport2.jpg"
image_path = "../dataset/data/exercise_image/form.jpg"
image = cv2.imread(image_path)

h, w, _ = image.shape
ry = h / input_size[1]
rx = w / input_size[0]

blob = cv2.dnn.blobFromImage(image, 1.0, input_size, (123.68, 116.78, 103.94), swapRB=True, crop=False)

model_path = "../dataset/data/text_detection/east_model/frozen_east_text_detection.pb"
net = cv2.dnn.readNet(model_path)

net.setInput(blob)
scores, geometry = net.forward(outBlobNames=["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
# scores 的形状为: (1, 1, 80, 80), geometry 的形状为: (1, 5, 80, 80). 输入图像大小为: (320, 320).
# EAST 算法将图像划分成 4*4 的小格, 并对每一格判断是否为文本.

_, _, rows, cols = scores.shape
rects = []
confidences = []

# 判断是否为文本的阈值.
min_confidence = 0.5
for i in range(rows):
    for j in range(cols):
        score = scores[0, 0, i, j]
        if score < min_confidence:
            continue
        # 计算当前单元格在 (320, 320) 图像中的位置.
        offset_x, offset_y = j * 4, i * 4

        # 提取, 预测出的文本的方向. 弦度.
        angle = geometry[:, 4, i, j]
        cos = np.cos(angle)
        sin = np.sin(angle)

        # 使用几何体体积导出边界框的宽度和高度.
        # l0: 表示文本框上端到 offset 锚点的距离,
        # l1: 表示文本框右端到 offset 锚点的距离,
        # l2: 表示文本框下端到 offset 锚点的距离,
        # l3: 表示文本框左端到 offset 锚点的距离,
        # angle: 表示文本框逆时针旋转的角度. 由于 cv2 没有画旋转矩形的方法, 以下按角度为 0 的方式画矩形框.
        l0 = geometry[0, 0, i, j]
        l1 = geometry[0, 1, i, j]
        l2 = geometry[0, 2, i, j]
        l3 = geometry[0, 3, i, j]

        h = l0 + l2
        w = l1 + l3

        # compute both the starting and ending (x, y)-coordinates for
        # the text prediction bounding box
        endX = int(offset_x + (cos * l1) + (sin * l2))
        endY = int(offset_y - (sin * l1) + (cos * l2))
        startX = int(endX - w)
        startY = int(endY - h)

        # add the bounding box coordinates and probability score to
        # our respective lists
        rects.append([startX, startY, endX, endY])
        confidences.append(score)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective
    # ratios
    startX = int(startX * rx)
    startY = int(startY * ry)
    endX = int(endX * rx)
    endY = int(endY * ry)

    # draw the bounding box on the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

show_image(image)
