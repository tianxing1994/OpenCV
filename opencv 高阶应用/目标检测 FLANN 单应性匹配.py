"""
说明:
最近在做一个需求, 是需要识别游戏界面中的 QQ/微信 登录上面的 QQ图标和微信图标.
这个如果像本例中用一个 QQ 图标上提取出来的特征, 是很难在所有界面有效的.
所以在这个需求中,
我标注了 100 款游戏的 QQ 和微信图标.
然后使用 SIFT 提取特征描述符 descriptors. 将其进行聚类得到 300 个描述符.
再用这些描述符去目标图片中匹配特征, 可以得到相匹配的特征点.
再用这些特征点的坐标去分析图标的位置.
这样只是不能再进行透视变换, 但是可以确定出图标的大概位置.

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


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


image1 = cv.imread('../dataset/data/other_sample/box.png',0)
image2 = cv.imread('../dataset/data/other_sample/box_in_scene.png', 0)

# detector = cv.xfeatures2d.SIFT_create()
detector = cv.xfeatures2d.SURF_create()
keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

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

show_image(image3)
