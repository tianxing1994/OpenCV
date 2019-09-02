"""
参考链接:
https://blog.csdn.net/tengfei461807914/article/details/80978947

LK 算法计算的是稀疏的特征点光流, 如样例当中计算的是使用 Shi-Tomasi 算法得到的特征点.
opencv当总提供了查找稠密光流的方法. 该方法计算一帧图像当中的所有点.

Farneback 稠密光流的主要思想是利用多项式对每个像素的邻域信息进行近似表示, 例如考虑二次多项式.
f(x) = x.TAx + b.Tx + c

"""

import numpy as np
import cv2 as cv

cap = cv.VideoCapture("C:/Users/Administrator/PycharmProjects/OpenCV/dataset/data/768x576.avi")

# 获取第一帧
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)

# 遍历每一行的第一列
hsv[:, :, 1] = 255

while True:
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # 返回一个两通道的光流向量, 实际上是每个点的像素位移值
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 笛卡尔坐标转换为极坐标, 获得极轴和极角
    mag, ang = cv.cartToPolar(flow[:, :, 0], flow[:, :, 1])

    hsv[:, :, 0] = ang * 180 / np.pi / 2
    hsv[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    cv.imshow("frame2", rgb)
    if cv.waitKey(30) == 27:
        break
    elif cv.waitKey(30) == ord('s'):
        cv.imwrite('opticalb.png', frame2)
        cv.imwrite('opticalhsv.png', rgb)
    prvs = next

cap.release()
cv.destroyAllWindows()

