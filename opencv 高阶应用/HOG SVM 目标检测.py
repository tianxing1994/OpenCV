"""
参考链接:
https://blog.csdn.net/hongbin_xu/article/details/79845290
https://github.com/ToughStoneX/hog_pedestran_detect_python
https://blog.csdn.net/qq_33662995/article/details/79356939

https://blog.csdn.net/WZZ18191171661/article/details/91305466
https://blog.csdn.net/yongjiankuang/article/details/79808346

说明:
1. 基本流程明确. 但是最终的效果极差, 且速度很慢.
2. 在以下链接中还有介绍一个过程:
https://blog.csdn.net/WZZ18191171661/article/details/91305466
将训练好的 SVM 模型应用于完全没有行人的图片中进行检测,
将其检测出的结果(难例)全部作为负例加入训练集中重新训练模型以提取准确率.
我尝试了, 目前的模型在训练集上的表现比较好, 也许是需要添加难例,
还有一个问题就是总感觉这个 detectMultiScale 只会在图像的中间检测一样.

3. 在 demo3 中我尝试加载保存起来的 SVM 参数. 但是后面的流程中出现报错.
4. 在 demo4 中我保存了已经设置好 SVM 检测器的 HOG 实例, 之后再加载已保存的实例进行行人检测. 程序可执行.
5. 总共的正负样本才 1848 个. 而我们的 HOG 特征描述符则有 3780 个维度. 这显然, 数据还远远不够.
任何一种机器学习都是通过在空间中对比测试样本的位置及其邻近的已知训练样本的类别来判断该测试样本的类别.
当维数很高, 而样本较少时, 则在空间中, 训练样本几乎不能保证填满每一个格子. 这显然不行.
6. 看起来由于训练样本的不足导致训练结果的检测准确率很差. 但如果我具有更多的样本, 我想,
则完全可以尝试使用深度学习的方法, 如此, 我觉得, 是不是传统的机器学习在这方面已经没有价值了呢.
"""
import os

import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def load_data(data_path):
    """
    加载数据, 生成训练集和测试集.
    正样本和负样本图片的形状为: (128, 64, 3). 测试集图片形状为: (480, 640, 3).
    :return:
    """
    positive_path = os.path.join(data_path, 'positive')
    negative_path = os.path.join(data_path, 'negative')
    test_data_path = os.path.join(data_path, 'test_data')

    image_list = list()
    label_list = list()
    test_list = list()

    for filename in os.listdir(positive_path):
        image_path = os.path.join(positive_path, filename)
        image = cv.imread(image_path)
        image_list.append(image)
        label_list.append(1)

    for filename in os.listdir(negative_path):
        image_path = os.path.join(negative_path, filename)
        image = cv.imread(image_path)
        image_list.append(image)
        label_list.append(0)

    for filename in os.listdir(test_data_path):
        image_path = os.path.join(test_data_path, filename)
        image = cv.imread(image_path)
        test_list.append(image)

    train = np.stack(image_list, axis=0)
    target = np.array(label_list)
    test = np.stack(test_list, axis=0)
    return train, target, test


def get_hog_detector():
    win_size = (64, 128)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    args = (win_size, block_size, block_stride, cell_size, nbins)
    hog = cv.HOGDescriptor(*args)
    return hog


def extract_hog_data(train):
    """
    将图片转换为 HOG 特征. 原图像是 BGR 三通道, 需传换为 Gray 灰度图.
    :param train:
    :return:
    """
    result = list()

    hog = get_hog_detector()

    for sample in train:
        gray = cv.cvtColor(sample, cv.COLOR_BGR2GRAY)
        # descriptors.shape = (3780, 1)
        descriptors = hog.compute(gray)
        result.append(descriptors)

    result = np.stack(result, axis=0)
    result = np.squeeze(result)
    return result


def get_svm_detector(svm):
    """
    从训练好的 SVM 分类器中取出支持向量和 rho 参数.
    导出可以用于 cv2.HOGDescriptor() 的 SVM 检测器, 实质上是训练好的 SVM 的支持向量和 rho 参数组成的列表.
    :param svm: 训练好的 SVM 分类器.
    :return: SVM 的支持向量和 rho 参数组成的列表, 可用作 cv2.HOGDescriptor() 的 SVM 检测器.
    """
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


def get_empty_svm():
    svm = cv.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)    # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)    # From paper, soft classifier
    svm.setType(cv.ml.SVM_EPS_SVR)
    return svm


def people_detect_by_image(image, hog):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # foundLocations, foundWeights = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
    foundLocations, foundWeights = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
    print("foundLocations: ", foundLocations)
    print("foundWeights: ", foundWeights)
    for (x, y, w, h), weight in zip(foundLocations, foundWeights):
        if weight > 0.1:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    show_image(image)
    return


def people_detect(test, hog):
    for image in test:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 我感觉这个 detectMultiScale 好像只会在图像中间去找.
        # foundLocations, foundWeights = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
        foundLocations, foundWeights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
        print("foundLocations: ", foundLocations)
        print("foundWeights: ", foundWeights)
        for (x, y, w, h), weight in zip(foundLocations, foundWeights):
            if weight > 0.1:
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        show_image(image)
    return


def demo1():
    """检测数据. """
    image_path = '../dataset/data/hog_svm_people_detection/positive/per00001.ppm'
    image = cv.imread(image_path)
    show_image(image)
    return


def demo2():
    """从加载数据, 训练 SVM, 到 HOG 行人检测. """
    data_path = '../dataset/data/hog_svm_people_detection'

    train, target, test = load_data(data_path)
    train = extract_hog_data(train)
    svm = get_empty_svm()
    svm.train(train, cv.ml.ROW_SAMPLE, target)
    svm_detector = get_svm_detector(svm)
    hog = get_hog_detector()
    hog.setSVMDetector(svm_detector)
    people_detect(test, hog)
    return


def demo3_1():
    """
    demo3_1 训练并保存 SVM 模型, 之后再加载进来使用.
    我以为这样可以, 但是实际上不行.
    """
    data_path = '../dataset/data/hog_svm_people_detection'
    model_save_dir = '../dataset/data/hog_svm_people_detection'
    model_path = os.path.join(model_save_dir, 'svm.xml')

    train, target, test = load_data(data_path)
    train = extract_hog_data(train)
    svm = get_empty_svm()
    svm.train(train, cv.ml.ROW_SAMPLE, target)

    svm.save(model_path)
    return


def demo3_2():
    """
    demo3_2 加载训练好的 SVM 模型进来使用.
    我以为这样可以, 但是实际上不行.
    """
    data_path = '../dataset/data/hog_svm_people_detection'
    model_save_dir = '../dataset/data/hog_svm_people_detection'
    model_path = os.path.join(model_save_dir, 'svm.xml')
    _, _, test = load_data(data_path)

    svm = get_empty_svm()
    svm.load(model_path)
    svm_detector = get_svm_detector(svm)    # 报错
    hog = get_hog_detector()
    hog.setSVMDetector(svm_detector)
    people_detect(test, hog)
    return


def demo4_1():
    """训练并保存 hog 模型. """
    data_path = '../dataset/data/hog_svm_people_detection'
    model_save_dir = '../dataset/data/hog_svm_people_detection'
    model_path = os.path.join(model_save_dir, 'hog.bin')

    train, target, test = load_data(data_path)

    train = extract_hog_data(train)
    svm = get_empty_svm()
    svm.train(train, cv.ml.ROW_SAMPLE, target)
    svm_detector = get_svm_detector(svm)
    hog = get_hog_detector()
    hog.setSVMDetector(svm_detector)
    hog.save(filename=model_path)
    return


def demo4_2():
    """加载已训练好的模型, 执行行人检测. """
    data_path = '../dataset/data/hog_svm_people_detection'

    model_save_dir = '../dataset/data/hog_svm_people_detection'

    model_path = os.path.join(model_save_dir, 'hog.bin')
    train, _, test = load_data(data_path)

    hog = get_hog_detector()
    hog.load(filename=model_path)

    people_detect(test, hog)
    return


def demo4_3():
    """加载已训练好的模型, 执行行人检测. """
    # image_path = '../dataset/data/hog_svm_people_detection/positive/per00120.ppm'
    # image_path = '../dataset/data/hog_svm_people_detection/negative/000009.jpg'
    # image_path = '../dataset/data/hog_svm_people_detection/test_data/000029.jpg'
    image_path = '../dataset/data/people/people3.jpg'
    image = cv.imread(image_path)

    model_save_dir = '../dataset/data/hog_svm_people_detection'
    model_path = os.path.join(model_save_dir, 'hog.bin')

    hog = get_hog_detector()
    hog.load(filename=model_path)

    people_detect_by_image(image, hog)
    return


def demo5():
    """
    opencv 自带的行人检测器.
    训练出来的模型, 运行速度慢, 而且准确性极差.
    """
    data_path = '../dataset/data/hog_svm_people_detection'
    _, _, test = load_data(data_path)
    hog = get_hog_detector()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    people_detect(test, hog)
    return


if __name__ == '__main__':
    # demo2()
    demo4_2()
    # demo4_3()
    # demo5()
