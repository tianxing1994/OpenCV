"""
参考链接:
https://blog.csdn.net/HuangZhang_123/article/details/80660688

"""
import numpy as np
import cv2 as cv


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def orb_match(template, scene):
    orb = cv.ORB_create(nfeatures=50)
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(scene, None)

    # 暴力匹配 BFMatcher, 遍历描述符, 确定描述符是否匹配, 然后计算匹配距离并排序
    # BFMatcher 函数参数:
    # normType: NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2.
    # NORM_L1 和 NORM_L2 是 SIFT 和 SURF 描述符的优先选择, NORM_HAMMING 和 NORM_HAMMING2 是用于 ORB 算法
    bf = cv.BFMatcher(normType=cv.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des1,des2)

    matches = sorted(matches, key=lambda x: x.distance)
    # matches是DMatch对象，具有以下属性：
    # DMatch.distance - 描述符之间的距离. 越低越好.
    # DMatch.trainIdx - 训练描述符中描述符的索引
    # DMatch.queryIdx - 查询描述符中描述符的索引
    # DMatch.imgIdx - 训练图像的索引.

    image3 = cv.drawMatches(img1=template, keypoints1=kp1, img2=scene, keypoints2=kp2, matches1to2=matches, outImg=scene, flags=2)
    show_image(image3)
    return


def orb_knn_match(template, scene):
    """
    在 tmeplate, scene 中提取 ORB 特征, 并进行 KNN 匹配, 展示效果.
    :param template: 目标图像
    :param scene: 场景图像
    :return:
    """
    orb = cv.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(template, None)
    keypoints2, descriptors2 = orb.detectAndCompute(scene, None)

    # 暴力匹配 BFMatcher, 遍历描述符, 确定描述符是否匹配, 然后计算匹配距离并排序
    # BFMatcher 函数参数:
    # normType：NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2。
    # NORM_L1 和 NORM_L2 是 SIFT 和 SURF 描述符的优先选择, NORM_HAMMING 和 NORM_HAMMING2 是用于 ORB 算法.
    bf = cv.BFMatcher(normType=cv.NORM_HAMMING, crossCheck=True)
    # knnMatch 函数参数 k 是返回符合匹配的个数，暴力匹配 match 只返回最佳匹配结果.
    matches = bf.knnMatch(descriptors1, descriptors2, k=1)

    # 使用drawMatchesKnn函数将结果显示
    image = cv.drawMatchesKnn(img1=template, keypoints1=keypoints1, img2=scene,
                              keypoints2=keypoints2, matches1to2=matches[:20], outImg=scene, flags=2)
    show_image(image)
    return


def demo1():
    # template = cv.imread('../dataset/data/other_sample/box.png', 0)
    # scene = cv.imread('../dataset/data/other_sample/box_in_scene.png', 0)
    template = cv.imread('../dataset/data/other_sample/python.jpg', 0)
    scene = cv.imread('../dataset/data/other_sample/python_in_scene.jpg', 0)
    orb_knn_match(template,scene)
    return


def demo2():
    template = cv.imread('../dataset/data/other_sample/python.jpg', 0)
    scene = cv.imread('../dataset/data/other_sample/python_in_scene.jpg', 0)
    orb_match(template, scene)
    return


if __name__ == '__main__':
    demo1()
