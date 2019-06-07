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


class Vertex(object):

    def __init__(self, id, x, y, value):
        self.id = id
        self.x = x
        self.y = y
        self.value = value
        self._parent = None
        self._original_parent = None
        self._size = 1

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value
        self.update_original_parent()
        return

    def update_original_parent(self):
        real_original_parent = self.find_original_parent()
        if self.parent is not real_original_parent:
            self.parent = real_original_parent
        return

    def find_original_parent(self, vertex=None):
        if vertex is None:
            vertex = self
        if vertex.parent is None:
            return self

        original_parent = vertex.parent
        while True:
            if original_parent.parent is None: break
            original_parent = original_parent.parent
        return original_parent

    @property
    def size(self):
        if self._original_parent is None:
            result = self._size
        else:
            result = self._original_parent.size
        return result

    def merge(self, other):
        if self._original_parent is other._original_parent:
            return
        other._original_parent.parent = self._original_parent
        return


class Edge(object):
    def __init__(self, left_vertex, right_vertex):
        self.left_vertex = left_vertex
        self.right_vertex = right_vertex
        self.weight = self.calculate_weight()

    def calculate_weight(self):
        _result = np.sum((self.left_vertex.value - self.right_vertex.value) ** 2)
        return np.sqrt(_result)


class Graph(object):
    def __init__(self, image_path):
        self.image = cv.imread(image_path)
        self.image_height = self.image[0]
        self.image_width = self.image[1]

        self.graph_vertex = None
        self.graph_edges = None

    def _init_vertex(self):
        self.vertex_no = self.image_height * self.image_width
        self.graph_vertex = []
        for y in range(self.image_height):
            for x in range(self.image_width):
                vertex_id = y * self.image_width + x
                value = self.image[y, x]
                self.graph_vertex.append(Vertex(vertex_id, x, y, value))
        return

    def _init_edges(self):
        self.graph_edges = []
        for vertex in self.graph_vertex:
            if vertex.x
        for y in range(self.image_height):
            for x in range(self.image_width):
                self.graph_edges.append(Edge(vertex_id, x, y, value))
        return

    def _difference(self, x1, y1, x2, y2):
        _result = np.sum((self.image[x1, y1] - self.image[x2, y2]) ** 2)
        return np.sqrt(_result)

    def _create_edge(self, x1, y1, x2, y2):
        vertex_id = lambda x, y: y * self.image_width + x
        w = self._difference(x1, y1, x2, y2)
        return (vertex_id(x1, y1), vertex_id(x2, y2), w)

    def build_graph(self, neighborhood_8=False):
        self.graph_edges = []
        for y in range(self.image_height):
            for x in range(self.image_width):
                if x > 0:
                    self.graph_edges.append(self._create_edge(x, y, x-1, y))
                if y > 0:
                    self.graph_edges.append(self._create_edge(x, y, x, y-1))
                if neighborhood_8:
                    if x > 0 and y > 0:
                        self.graph_edges.append(self._create_edge(x, y, x-1, y-1))
                    if x > 0 and y < self.image_height-1:
                        self.graph_edges.append(self._create_edge(x, y, x-1, y+1))
        return self.graph_edges


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

    return forest


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


import argparse
import logging
import time
from random import random
from PIL import Image, ImageFilter
from skimage import io
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
    show_image(image)
    return image


def get_segmented_image(sigma, neighbor, K, min_comp_size, input_file, output_file):
    if neighbor != 4 and neighbor != 8:
        logger.warn('Invalid neighborhood choosed. The acceptable values are 4 or 8.')
        logger.warn('Segmenting with 4-neighborhood...')
    start_time = time.time()
    image_file = Image.open(input_file)

    size = image_file.size  # (width, height) in Pillow/PIL
    logger.info('Image info: {} | {} | {}'.format(image_file.format, size, image_file.mode))

    # Gaussian Filter
    smooth = image_file.filter(ImageFilter.GaussianBlur(sigma))
    smooth = np.array(smooth)

    logger.info("Creating graph...")
    graph_edges = build_graph(smooth, size[1], size[0], diff, neighbor == 8)

    logger.info("Merging graph...")
    forest = segment_graph(graph_edges, size[0] * size[1], K, min_comp_size, threshold)

    logger.info("Visualizing segmentation and saving into: {}".format(output_file))
    image = generate_image(forest, size[1], size[0])
    image.save(output_file)

    logger.info('Number of components: {}'.format(forest.num_sets))
    logger.info('Total running time: {:0.4}s'.format(time.time() - start_time))


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='Graph-based Segmentation')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='a float for the Gaussin Filter')
    parser.add_argument('--neighbor', type=int, default=8, choices=[4, 8],
                        help='choose the neighborhood format, 4 or 8')
    parser.add_argument('--K', type=float, default=10.0,
                        help='a constant to control the threshold function of the predicate')
    parser.add_argument('--min-comp-size', type=int, default=2000,
                        help='a constant to remove all the components with fewer number of pixels')
    # ../dataset/other/people.jpg; ../dataset/other/panda.jpg
    parser.add_argument('--input-file', type=str, default="../dataset/other/people.jpg",
                        help='the file path of the input image')
    # ../dataset/other/people_output.jpg; ../dataset/other/panda_output.jpg
    parser.add_argument('--output-file', type=str, default="../dataset/other/people_output.jpg",
                        help='the file path of the output image')
    args = parser.parse_args()

    # basic logging settings
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    logger = logging.getLogger(__name__)

    get_segmented_image(args.sigma, args.neighbor, args.K, args.min_comp_size, args.input_file, args.output_file)


