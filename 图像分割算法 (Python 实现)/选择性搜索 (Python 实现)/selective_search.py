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


from random import random, randrange
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


def generate_label_map(forest, width, height):
    random_label = lambda: randrange(width * height)
    colors = [random_label() for i in range(width * height)]

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
    # image = generate_image(forest, size[1], size[0])
    image = generate_label_map(forest, size[1], size[0])
    image = np.array(image)
    # show_image(image)
    return image


# <! 以上是 graphbases_image_segmentation. 以下是 selective search. >
import numpy
import skimage
import skimage.io
import skimage.feature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def _generate_segments(img_path, neighbor, sigma, scale, min_size):
    # open the Image
    # im_mask = graphbased_segmentation(img_path, neighbor, sigma, scale, min_size)
    im_mask = graphbased_segmentation(sigma, neighbor, scale, min_size, img_path)
    im_orig = skimage.io.imread(img_path)
    # merge mask channel to the image as a 4th channel
    im_orig = numpy.append(
        im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)

    im_orig[:, :, 3] = im_mask[:, :, 0]
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


def _sim_colour(r1, r2):
    """
        calculate the sum of histogram intersection of colour
    """
    # return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])
    return sum([1 if a==b else 1-float(abs(a - b))/max(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])/len(r1)


def _sim_texture(r1, r2):
    """
        calculate the sum of histogram intersection of texture
    """
    # return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])
    return sum([1 if a==b else 1-float(abs(a - b))/max(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])/len(r1)


def _sim_size(r1, r2, imsize):
    """
        calculate the size similarity over the image
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def _calc_sim(r1, r2, imsize):
    return (_sim_colour(r1, r2) + _sim_texture(r1, r2)
            + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize))


def _extract_regions(img):
    # 创建字典
    R = {}
    # get hsv image
    hsv = skimage.color.rgb2hsv(img[:, :, :3])

    # pass 1: count pixel positions
    # 遍历img中所有的元素，y为索引，i为一个（r,g,b,l）
    for y, i in enumerate(img):
        for x, (r, g, b, l) in enumerate(i):
            # initialize a new region
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}
            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

    # pass 2: calculate texture gradient
    tex_grad = _calc_texture_gradient(img)

    # pass 3: calculate colour histogram of each region
    for k, v in list(R.items()):
        # colour histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)
        # texture histogram
        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])

    return R


def _extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    R = list(regions.items())
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours


def _merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


def selective_search(
        img_path, neighbor, sigma, scale, min_size):

    # load image and get smallest regions
    # region label is stored in the 4th value of each pixel [r,g,b,(region)]
    img = _generate_segments(img_path, neighbor, sigma, scale, min_size)

    if img is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]
    R = _extract_regions(img)
    # print(R[0])

    # extract neighbouring information
    neighbours = _extract_neighbours(R)
    # print(neighbours[0])
    # calculate initial similarities
    # 创建字典
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        # print(ai)
        # print(bi)
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    # hierarchal search
    while S != {}:

        # get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # merge corresponding regions
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])

        # mark similarities for regions to be removed
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                # 去除这两个区域与相邻区域的相似度
                key_to_delete.append(k)

        # remove old similarities of related regions
        for k in key_to_delete:
            del S[k]

        # calculate similarity set with the new region
        # 计算合并后区域与相邻区域的相似度
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)

    regions = []
    for k, r in list(R.items()):
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    return img, regions


def main():

    img_path = "../../dataset/other/panda.jpg"
    # loading astronaut image
    img = skimage.io.imread(img_path)

    # perform selective search
    img_lbl, regions = selective_search(
        img_path, neighbor = 8 , sigma = 0.5, scale = 200, min_size = 20)

    # 创建集合candidate
    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    main()




