"""
参考链接:
https://blog.csdn.net/u014796085/article/details/83478583
https://blog.csdn.net/mao_kun/article/details/50576003

1. 使用 Efficient Graph-Based Image Segmentation 的方法获取原始分割区域 R={r1,r2,…,rn}.
2. 初始化相似度集合 S=∅
3. 计算两两相邻区域之间的相似度, 将其添加到相似度集合 S 中
4. 从相似度集合 S 中找出, 相似度最大的两个区域 ri 和 rj, 将其合并成为一个区域 rt,
从相似度集合中除去原先与 ri 和 rj 相邻区域之间计算的相似度, 计算 rt 与其相邻区域 (原先与 ri 或 rj 相邻的区域) 的相似度,
将其结果添加的到相似度集合 S 中. 同时将新区域 rt 添加区域集合 R 中.
5. 重复步骤 4, 直到 S=∅, 即最后一个新区域 rt 为整幅图像.
6. 获取 R 中每个区域的 Bounding Boxes, 去除像素数量小于 2000, 以及宽高比大于 1.2 的, 剩余的框就是物体位置的可能结果 L.
"""
"""
基于图的图像分割Effective graph-based image segmentation
https://blog.csdn.net/u014796085/article/details/83449972
https://blog.csdn.net/surgewong/article/details/39008861
github.com/luisgabriel/image-segmentation


1. 首先, 将图像(image)表达成图论中的图(graph).
具体说来就是, 把图像中的每一个像素点看成一个顶点 vi∈V (node 或 vertex),
每个像素与相邻 8 个像素 (8-邻域) 构成图的一条边 ei∈E, 这样就构建好了一个图 G = (V,E).
图每条边的权值是像素与相邻像素的关系 (灰度图的话是灰度值差的绝对值, RGB图像为3个通道值差平方和开根号),
表达了相邻像素之间的相似度.

2. 将每个节点 (像素点) 看成单一的区域, 然后进行合并.
(1) 对所有边根据权值从小到大排序, 权值越小, 两像素的相似度越大.
(2) S[0] 是一个原始分割, 相当于每个顶点当做是一个分割区域.
(3) 从小到大遍历所有边, 如果这条边 (vi,vj) 的两个顶点属于不同的分割区域,
并且权值不大于两个区域的内部差(区域内左右边最大权值), 那么合并这两个区域. 更新合并区域的参数和内部差.
因为遍历时是从小到大遍历, 所以如果合并, 这条边的权值一定是新区域所有边最大的权值.

3. 最后对所有区域中，像素数都小于min_size的两个相邻区域，进行合并得到最后的分割。
"""

class Node:
    def __init__(self, parent, rank=0, size=1):
        self.parent = parent
        self.rank = rank
        self.size = size

    def __repr__(self):
        return '(parent=%s, rank=%s, size=%s)' % (self.parent, self.rank, self.size)


class Forest:
    def __init__(self, num_nodes):
        self.nodes = [Node(i) for i in range(num_nodes)]
        self.num_sets = num_nodes

    def size_of(self, i):
        return self.nodes[i].size

    def find(self, n):
        """
        :param n: Node 节点对象.
        :return: 返回 n 在 self.nodes 列表中对应的 Node 对象的最终父节点.
        """
        temp = n
        while temp != self.nodes[temp].parent:
            temp = self.nodes[temp].parent

        self.nodes[n].parent = temp
        return temp

    def merge(self, a, b):
        if self.nodes[a].rank > self.nodes[b].rank:
            self.nodes[b].parent = a
            self.nodes[a].size = self.nodes[a].size + self.nodes[b].size
        else:
            self.nodes[a].parent = b
            self.nodes[b].size = self.nodes[b].size + self.nodes[a].size

            if self.nodes[a].rank == self.nodes[b].rank:
                self.nodes[b].rank = self.nodes[b].rank + 1

        self.num_sets = self.num_sets - 1

    def print_nodes(self):
        for node in self.nodes:
            print(node)


def create_edge(img, width, x, y, x1, y1, diff):
    # 从左向右, 从上向下扫描图像, (x, y) 像素点是第 y*width+x 个像素点. width 是图像的宽度.
    vertex_id = lambda x, y: y * width + x
    w = diff(img, x, y, x1, y1)
    return (vertex_id(x, y), vertex_id(x1, y1), w)


def build_graph(img, width, height, diff, neighborhood_8=False):
    graph_edges = []
    for y in range(height):
        for x in range(width):
            if x > 0:
                graph_edges.append(create_edge(img, width, x, y, x-1, y, diff))

            if y > 0:
                graph_edges.append(create_edge(img, width, x, y, x, y-1, diff))

            if neighborhood_8:
                if x > 0 and y > 0:
                    graph_edges.append(create_edge(img, width, x, y, x-1, y-1, diff))

                if x > 0 and y < height-1:
                    graph_edges.append(create_edge(img, width, x, y, x-1, y+1, diff))

    return graph_edges


def remove_small_components(forest, graph, min_size):
    """
    遍历 graph_edges. 将较小的块合并为一个.
    :param forest:
    :param graph:
    :param min_size:
    :return:
    """
    for edge in graph:
        a = forest.find(edge[0])
        b = forest.find(edge[1])

        if a != b and (forest.size_of(a) < min_size or forest.size_of(b) < min_size):
            forest.merge(a, b)

    return  forest


def segment_graph(graph_edges, num_nodes, const, min_size, threshold_func):
    # Step 1: initialization
    forest = Forest(num_nodes)
    weight = lambda edge: edge[2]
    sorted_graph = sorted(graph_edges, key=weight)
    threshold = [ threshold_func(1, const) for _ in range(num_nodes) ]

    # Step 2: merging
    for edge in sorted_graph:
        parent_a = forest.find(edge[0])
        parent_b = forest.find(edge[1])
        a_condition = weight(edge) <= threshold[parent_a]
        b_condition = weight(edge) <= threshold[parent_b]

        if parent_a != parent_b and a_condition and b_condition:
            forest.merge(parent_a, parent_b)
            a = forest.find(parent_a)
            threshold[a] = weight(edge) + threshold_func(forest.nodes[a].size, const)

    return remove_small_components(forest, sorted_graph, min_size)


from random import random
from PIL import Image, ImageFilter
import numpy as np
import cv2 as cv


def diff(img, x1, y1, x2, y2):
    _out = np.sum((img[x1, y1] - img[x2, y2]) ** 2)
    return np.sqrt(_out)


def threshold(size, const):
    return (const * 1.0 / size)


def generate_image(forest, width, height):
    random_color = lambda: (int(random() * 255), int(random() * 255), int(random() * 255))
    colors = [random_color() for i in range(width * height)]

    img = Image.new('RGB', (width, height))
    im = img.load()
    for y in range(height):
        for x in range(width):
            comp = forest.find(y * width + x)
            im[x, y] = colors[comp]

    return img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def graphbased_segmentation(sigma, neighbor, K, min_comp_size, input_file):
    image_file = Image.open(input_file)
    size = image_file.size
    smooth = image_file.filter(ImageFilter.GaussianBlur(sigma))
    smooth = np.array(smooth)
    graph_edges = build_graph(smooth, size[1], size[0], diff, neighbor == 8)
    forest = segment_graph(graph_edges, size[0] * size[1], K, min_comp_size, threshold)
    image = generate_image(forest, size[1], size[0])
    image = np.array(image)
    # show_image(image)
    return image


# <! 以上是 graphbases_image_segmentation. 以下是 selective search. >
import numpy
import skimage
import skimage.io
import skimage.feature


def _generate_segments(img_path, neighbor, sigma, scale, min_size):
    # open the Image
    im_mask = graphbased_segmentation(img_path, neighbor, sigma, scale, min_size)
    im_orig = skimage.io.imread(img_path)
    # merge mask channel to the image as a 4th channel
    im_orig = numpy.append(
        im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask
    return im_orig


def _calc_colour_hist(img):
    """
        calculate colour histogram for each region
        the size of output histogram will be BINS * COLOUR_CHANNELS(3)
        number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
        extract HSV
    """
    BINS = 25
    hist = numpy.array([])
    for colour_channel in (0, 1, 2):
        # extracting one colour channel
        c = img[:, colour_channel]
        # calculate histogram for each colour and join to the result
        hist = numpy.concatenate(
            # result of np.histogram like to: (array([......], dtype=int64), array([ ......]))
            [hist] + [numpy.histogram(c, BINS, (0.0, 255.0))[0]])
    # L1 normalize
    hist = hist / len(img)
    return hist


def _calc_texture_gradient(img):
    """
        calculate texture gradient for entire image
        The original SelectiveSearch algorithm proposed Gaussian derivative
        for 8 orientations, but we use LBP instead.
        output will be [height(*)][width(*)]
    """
    ret = numpy.zeros(img.shape)

    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)
    return ret


def _calc_texture_hist(img):
    """
        calculate texture histogram for each region

        calculate the histogram of gradient for each colours
        the size of output histogram will be
            BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """
    BINS = 10
    hist = numpy.array([])
    for colour_channel in (0, 1, 2):
        # mask by the colour channel
        fd = img[:, colour_channel]
        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(fd, BINS, (0.0, 1.0))[0]])
    # L1 Normalize
    hist = hist / len(img)
    return hist


if __name__ == '__main__':
    image_path = '../../dataset/lena.png'
    image = cv.imread(image_path)
    result = _calc_texture_gradient(image)
    print(result)
    print(result.shape)
    print(len(result))




