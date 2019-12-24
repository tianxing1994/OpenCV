"""
参考链接:
https://blog.csdn.net/qq878594585/article/details/81901703
https://www.cnblogs.com/skyfsm/p/7411961.html
"""
# coding: utf-8
import numpy as np
import cv2

leftgray = cv2.imread('../dataset/data/image_stitching/shanghai_l.jpg')
rightgray = cv2.imread('../dataset/data/image_stitching/shanghai_r.jpg')

hessian = 400
surf = cv2.xfeatures2d.SURF_create(hessian)  # 将Hessian Threshold设置为400,阈值越大能检测的特征就越少
kp1, des1 = surf.detectAndCompute(leftgray, None)  # 查找关键点和描述符
kp2, des2 = surf.detectAndCompute(rightgray, None)

FLANN_INDEX_KDTREE = 0  # 建立FLANN匹配器的参数
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 配置索引，密度树的数量为5
searchParams = dict(checks=50)  # 指定递归次数
# FlannBasedMatcher：是目前最快的特征匹配算法（最近邻搜索）
flann = cv2.FlannBasedMatcher(indexParams, searchParams)  # 建立匹配器
# 在 trainDescriptors 中找两个最佳匹配给 queryDescriptors.
matches = flann.knnMatch(des1, des2, k=2)

good = []
# 提取优秀的特征点
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
        good.append(m)
src_pts = np.array([kp1[m.queryIdx].pt for m in good])
dst_pts = np.array([kp2[m.trainIdx].pt for m in good])
# 找到将 src_pts 变换到 dst_pts 的透视变换矩阵 M.
H = cv2.findHomography(src_pts, dst_pts)


h, w = leftgray.shape[:2]
h1, w1 = rightgray.shape[:2]
# 平移矩阵, 在对 left 图像做变换后将图像向右平移 w 距离.
shft = np.array([[1.0, 0, w],
                 [0, 1.0, 0],
                 [0, 0, 1.0]])

M = np.dot(shft, H[0])
dst_corners = cv2.warpPerspective(leftgray, M, (w * 2, h))  # 透视变换，新图像可容纳完整的两幅图
cv2.imshow('tiledImg1', dst_corners)  # 显示，第一幅图已在标准位置
dst_corners[0:h, w:w * 2] = rightgray  # 将第二幅图放在右侧
# cv2.imwrite('tiled.jpg',dst_corners)
cv2.imshow('tiledImg', dst_corners)
cv2.imshow('leftgray', leftgray)
cv2.imshow('rightgray', rightgray)
cv2.waitKey()
cv2.destroyAllWindows()
