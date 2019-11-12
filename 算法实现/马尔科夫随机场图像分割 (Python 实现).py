"""
参考链接:
https://blog.csdn.net/weixin_41923961/article/details/80878036
https://blog.csdn.net/m0_37637511/article/details/87966356

计算步骤:
1. 使用随机, 或 kmeans 等其他算法给图像中的各像素打上分类标签.
2. 对于图像中的各像素点, 我们根据其周围8邻域像素所属的标签来更新该像素点的标签. 即: 计算其8邻域中各标签出现的次数, 计算其属于各标签的概率 p_c
3. 计算各个分类标签的像素的平均值. 以及方差. 用以构建正态分布的高斯概率密度函数.
4. 对于图像中的各像素点, 根据正太分布函数, 及该像素的灰度值, 计算其属于该标签的概率 p_sc.
5. 通过 p_c, p_sc 相乘, 得出当前像素点属于各标签的概率. (实际计算过程中对概率值取对数, 用于计算)
6. 已计算出各像素点属于各标签的概率. 此取概率最大的标签做为各像素的新标签.
7. 迭代 2-6 步. 得出最终分割图像.
"""
import cv2 as cv
import numpy as np


def show_image(image):
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


image_path = '../dataset/data/image_sample/image0.JPG'
image = cv.imread(image_path)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray_d = np.array(gray, dtype=np.float64)

n_cluster = 3
max_iter = 200

label = np.random.randint(0, n_cluster, size=gray_d.shape)

f_u = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
f_d = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
f_l = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
f_r = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
f_ul = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
f_ur = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
f_dl = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
f_dr = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

current_iter = 0
while current_iter < max_iter:
    current_iter += 1
    label_u = cv.filter2D(np.array(label, dtype=np.uint8), -1, f_u)
    label_d = cv.filter2D(np.array(label, dtype=np.uint8), -1, f_d)
    label_l = cv.filter2D(np.array(label, dtype=np.uint8), -1, f_l)
    label_r = cv.filter2D(np.array(label, dtype=np.uint8), -1, f_r)
    label_ul = cv.filter2D(np.array(label, dtype=np.uint8), -1, f_ul)
    label_ur = cv.filter2D(np.array(label, dtype=np.uint8), -1, f_ur)
    label_dl = cv.filter2D(np.array(label, dtype=np.uint8), -1, f_dl)
    label_dr = cv.filter2D(np.array(label, dtype=np.uint8), -1, f_dr)

    m, n = label.shape
    p_c = np.zeros(shape=(n_cluster, m, n))

    # 计算对应像素点的八邻域中各标签出现的概率.
    for i in range(n_cluster):
        label_i = i * np.ones(shape=(m, n))
        u_t = np.array(label_i == label_u, dtype=np.int)
        d_t = np.array(label_i == label_d, dtype=np.int)
        l_t = np.array(label_i == label_l, dtype=np.int)
        r_t = np.array(label_i == label_r, dtype=np.int)
        ul_t = np.array(label_i == label_ul, dtype=np.int)
        ur_t = np.array(label_i == label_ur, dtype=np.int)
        dl_t = np.array(label_i == label_dl, dtype=np.int)
        dr_t = np.array(label_i == label_dr, dtype=np.int)

        count = u_t + d_t + l_t + r_t + ul_t + ur_t + dl_t + dr_t
        p_c[i, :] = count / 8

    # 由于后面要对概率进行对数变换, 所以不能有 0 值.
    p_c[p_c == 0] = 0.001

    mu = np.zeros(shape=(1, n_cluster))
    sigma = np.zeros(shape=(1, n_cluster))
    for i in range(n_cluster):
        data_c = gray_d[label == i]
        mu[0, i] = np.mean(data_c)
        sigma[0, i] = np.var(data_c)

    p_sc = np.zeros(shape=(n_cluster, m, n))
    one_a = np.ones(shape=(m, n))

    for j in range(n_cluster):
        mu_ = mu[0, j] * one_a
        # 正态分布函数公式. sigma 为方差, 即: σ^2. p_sc 中存储的是对应像素的灰度值属于各类别的概率
        p_sc[j, :] = (1.0 / np.sqrt(2 * np.pi * sigma[0, j])) * np.exp(-1 * ((gray_d - mu_)**2) / (2 * sigma[0, j]))

    # p_c, p_sc. 都代表的是概率. 即属于 0-1 之间. 两个小于 1 的数相乘, 可能会得到一个非常小的值,
    # 计算机对小数不能准确地保存, 所以这里使用 log 对数, 将其转换为更大的数, 会好一些.
    x_out = np.log(p_c) + np.log(p_sc)
    label_c = x_out.reshape(n_cluster, m*n)
    label_c_t = label_c.T
    label_m = np.argmax(label_c_t, axis=1)
    label = label_m.reshape(m, n)

label_w = label * (255 / (n_cluster-1))

show_image(label_w)
