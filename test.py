"""
这一份, 是我自己整理实现的. 在顶点合并时, 的阈值选取, 略有不同.

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
import cv2 as cv
import numpy as np
import random


class Node(object):
    def __init__(self, id, value):
        self.id = id
        self.value = value
        self.root = self.id
        self.size = 1


class Edge(object):
    def __init__(self, node1, node2, weight):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight


class Graph(object):
    def __init__(self, image_path, neighborhood_8=False, min_size=200):
        self.image = cv.imread(image_path)
        self.image_height = int(self.image.shape[0])
        self.image_width = int(self.image.shape[1])

        self.neighborhood_8 = neighborhood_8
        self.min_size = min_size

        self.nodes = self.create_nodes()
        self.edges = self.create_edges()

        self._label_image = None
        self._label_no = None
        self._segmented_image = None

    def create_nodes(self):
        nodes = []
        for y in range(self.image_height):
            for x in range(self.image_width):
                nodes.append(Node(y * self.image_width + x, self.image[y, x]))
        return nodes

    def diff(self, node1, node2):
        value1 = self.nodes[node1].value
        value2 = self.nodes[node2].value
        result = np.sqrt(np.sum((value1 - value2) ** 2))
        return result

    def create_edges(self):
        edges = []
        for y in range(self.image_height):
            for x in range(self.image_width):
                node1 = y * self.image_width + x
                if x > 0:
                    node2 = y * self.image_width + (x-1)
                    weight = self.diff(node1, node2)
                    edges.append(Edge(node1, node2, weight))
                if y > 0:
                    node2 = (y-1) * self.image_width + x
                    weight = self.diff(node1, node2)
                    edges.append(Edge(node1, node2, weight))
                if self.neighborhood_8:
                    if x > 0 and y > 0:
                        node2 = (y-1) * self.image_width + (x-1)
                        weight = self.diff(node1, node2)
                        edges.append(Edge(node1, node2, weight))
                    if x > 0 and y < self.image_height - 1:
                        node2 = (y+1) * self.image_width + (x-1)
                        weight = self.diff(node1, node2)
                        edges.append(Edge(node1, node2, weight))
        result = sorted(edges, key=lambda edge: edge.weight)
        return result

    # def threshold(self, size, const=10):
    #     return const * 1.0 / size

    def threshold(self, root1, root2):

        return 10

    def find_root(self, node):
        root = node
        while root != self.nodes[root].root:
            root = self.nodes[root].root
        return root

    def segment_graph(self):
        for edge in self.edges:
            root1 = self.find_root(edge.node1)
            root2 = self.find_root(edge.node2)
            if root1 != root2:
                if edge.weight < self.threshold(root1, root2):
                    self.merge(root1, root2)
        return

    def merge_small_components(self):
        for edge in self.edges:
            root1 = self.find_root(edge.node1)
            root2 = self.find_root(edge.node2)
            if root1 != root2:
                if self.nodes[root1].size < self.min_size or self.nodes[root2].size < self.min_size:
                    self.merge(root1, root2)
        return

    def merge(self, root1, root2):
        if root1 < root2:
            self.nodes[root2].root = root1
            self.nodes[root1].size = self.nodes[root1].size + self.nodes[root2].size
        if root1 > root2:
            self.nodes[root1].root = root2
            self.nodes[root2].size = self.nodes[root1].size + self.nodes[root2].size

    def generate_label_image(self):
        root_dict = dict()
        next_label = 0
        self.segment_graph()
        self.merge_small_components()
        _label_image = np.zeros(shape=self.image.shape[:2], dtype=np.uint8)
        for y in range(self.image_height):
            for x in range(self.image_width):
                index = y * self.image_width + x
                root = self.find_root(index)
                if root not in root_dict:
                    root_dict[root] = next_label
                    _label_image[y, x] = next_label
                    next_label += 1
                else:
                    the_label = root_dict[root]
                    _label_image[y, x] = the_label
        self._label_image = _label_image
        self._label_no = next_label
        return self._label_image, self._label_no

    def random_color(self):
        result = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return result

    def generate_segmented_image(self):
        _label_image, _label_no = self.generate_label_image()
        color_list = [self.random_color() for _ in range(_label_no)]
        _segmented_image = np.zeros(shape=self.image.shape, dtype=np.uint8)
        for y in range(self.image_height):
            for x in range(self.image_width):
                _label = _label_image[y, x]
                color = color_list[_label]
                _segmented_image[y, x] = color
        self._segmented_image = _segmented_image
        return self._segmented_image

    @property
    def label_image(self):
        if self._label_image is not None:
            return self._label_image
        _label_image, _label_no = self.generate_label_image()
        return _label_image

    @property
    def label_no(self):
        if self._label_no is not None:
            return self._label_no
        _label_image, _label_no = self.generate_label_image()
        return _label_no

    @property
    def segmented_image(self):
        if self._segmented_image is not None:
            return self._segmented_image
        _segmented_image = self.generate_segmented_image()
        return _segmented_image


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


if __name__ == '__main__':
    pass
    g = Graph(image_path='../dataset/other/phone.jpg')
    show_image(g.segmented_image)


