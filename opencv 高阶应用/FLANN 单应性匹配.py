"""
相关函数:
cv2.xfeatures2d.SIFT_create
sift.detectAndCompute
cv2.FlannBasedMatcher
flann.knnMatch
cv2.findHomography
numpy.float32
cv2.perspectiveTransform
cv2.polylines
cv2.drawMatches
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


image1 = cv.imread('../dataset/data/other_sample/box.png',0)
image2 = cv.imread('../dataset/data/other_sample/box_in_scene.png', 0)

sift = cv.xfeatures2d.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

# 指定 k=2, 为每一个 queryDescriptors 查找两个最佳匹配.
matches = flann.knnMatch(queryDescriptors=descriptors1,
                         trainDescriptors=descriptors2,
                         k=2)

# m 是最佳匹配, n 是次佳匹配, m.distance<n.distance. 那么, 当 m 的距离比 n*0.7 还要小时,
# 则认为这个 queryDescriptors 查询描述符具有一个可靠的匹配. 这是一个好的特征点.
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# 如果至少找到了 10 个优质的匹配.
if len(good) > 10:
    # 索引出每个匹配的 keypoint, 并获取其在图像中的坐标. 并整理成形如:
    # [[[6.8945546   6.163357]]
    #  [[34.13517   126.63356]]
    #           ...
    #  [[57.72485   129.60405]]
    #  [[69.18514    79.62016]]]
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 找到两个平面之间的透视变换.
    M, mask = cv.findHomography(srcPoints=src_points,
                                dstPoints=dst_points,
                                method=cv.RANSAC,
                                ransacReprojThreshold=5.0)


    matchesMask = mask.ravel().tolist()
    h, w = image1.shape
    points = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    # 使用转换矩阵将 image1 的大小映射到 image2 中, 再使用 ploylines 画直线.
    dst = cv.perspectiveTransform(points, M)
    image2 = cv.polylines(image2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(good), 10))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)

image3 = cv.drawMatches(image1, keypoints1, image2, keypoints2, good, None, **draw_params)

plt.imshow(image3, 'gray')
plt.show()

