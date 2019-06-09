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

    def threshold(self):
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
                if edge.weight < self.threshold():
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


# <! 以上是 graphbased_image_segmentation. 以下是 selective search. >
import skimage
import skimage.io
import skimage.feature


class SelectiveSearch(object):
    def __init__(self, image_path, gbis_neighborhood_8, gbis_min_size, lbp_radius=1.0, lbp_method='default'):
        self.image_path = image_path
        self.image = cv.imread(image_path)
        self.neighborhood_8 = gbis_neighborhood_8
        self.min_size = gbis_min_size
        self.image_size = self.image.shape[0] * self.image.shape[1]

        self._image_l = None
        self._lbp_image = None
        self._region_dict = None
        self._label_neighbor = None
        self._selective_search_box = None
        self._selective_search_image = None

        self.lbp_radius = lbp_radius
        self.lbp_n_points = 8 * self.lbp_radius
        self.lbp_method = lbp_method

    def _generate_segments(self):
        """
        读取图像, 将 graphbased image segmentation 算法得到的标签图像添加到图像的第 4 通道.
        :return:
        """
        g = Graph(self.image_path, self.neighborhood_8, self.min_size)
        label_image = g.label_image
        image_l = np.append(self.image, label_image[:, :, np.newaxis], axis=2)
        return image_l

    @property
    def image_l(self):
        if self._image_l is not None:
            return self._image_l
        else:
            self._image_l = self._generate_segments()
        return self._image_l

    def _calc_texture_gradient_image(self, n_points=None, radius=None, method=None):
        """
        计算图像的纹理图像.
        :return:
        """
        if n_points is None:
            n_points = self.lbp_n_points
        else:
            self.lbp_n_points = n_points

        if radius is None:
            radius = self.lbp_radius
        else:
            self.lbp_radius = radius

        if method is None:
            method = self.lbp_method
        else:
            self.lbp_method = method

        result = np.zeros(self.image.shape, dtype=np.int)
        for colour_channel in (0, 1, 2):
            result[:, :, colour_channel] = skimage.feature.local_binary_pattern(self.image[:, :, colour_channel], n_points, radius, method)
        return result

    @property
    def lbp_image(self):
        if self._lbp_image is not None:
            return self._lbp_image
        else:
            self._lbp_image = self._calc_texture_gradient_image()
        return self._lbp_image

    def _calc_colour_hist(self, pixel_list, bins=25):
        # pixel_list 是包含许多像素点的列表, 每个像素点是三个值的 ndarray. 它们都属于同一个 label 标签.
        hist = np.array([])
        for colour_channel in (0, 1, 2):
            c = pixel_list[:, colour_channel]
            hist = np.concatenate([hist] + [np.histogram(c, bins, (0.0, 255.0))[0]])
        hist = hist / len(pixel_list)
        return hist

    def _calc_texture_hist(self, pixel_list, bins=10):
        # pixel_list 是包含许多像素点的列表, 每个像素点是三个值的 ndarray. 它们具有相同的 label 标签.
        # 不过这里的每个像素上的值, 是 lbp 特征值.
        hist = np.array([])
        for colour_channel in (0, 1, 2):
            fd = pixel_list[:, colour_channel]
            hist = np.concatenate([hist] + [np.histogram(fd, bins, (0.0, pow(2, self.lbp_n_points) - 1))[0]])
        hist = hist / len(pixel_list)
        return hist

    def _sim_colour(self, r1, r2):
        # 给定两个块, r1, r2, 计算其颜色直方图中, 每一个 bin 上,
        # 来自 r1, r2 中, 较小值除以较大值. 以此判断颜色的相似度.
        # r1, r2 的长度应该都是 75.
        return sum([1 if a==b else float(min(a, b)/max(a, b)) for a, b in zip(r1["hist_c"], r2["hist_c"])])/len(r1)

    def _sim_texture(self, r1, r2):
        # 给定两个块, r1, r2, 计算其 lbp 纹理直方图中, 每一个 bin 上,
        # 来自 r1, r2 中, 较小值除以较大值. 以此判断纹理直方图的相似度.
        # r1, r2 的长度应该都是 75.
        return sum([1 if a==b else float(min(a, b)/max(a, b)) for a, b in zip(r1["hist_t"], r2["hist_t"])])/len(r1)

    def _sim_size(self, r1, r2):
        """
        记算大小相似度, 值越大越相似.
        """
        return 1.0 - (r1["size"] + r2["size"]) / self.image_size

    def _sim_fill(self, r1, r2):
        """
        calculate the fill similarity over the image
        """
        # bbsize, 两个块被识作一个块之后, 其 bounding box 的面积.
        bbsize = (
            (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
            * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
        )
        # 值越大, 越相似.
        return 1.0 - (bbsize - r1["size"] - r2["size"]) / self.image_size

    def _calc_sim(self, r1, r2):
        # 计算相似度.
        return (self._sim_colour(r1, r2) + self._sim_texture(r1, r2)
                + self._sim_size(r1, r2) + self._sim_fill(r1, r2))

    def _extract_regions(self):
        """
        根据 label 标记, 提取图像中的每一个块, 并存储到 region_dict.
        遍历图像, key 作为标签值, value 为:
        {"min_x": int, "min_y": int,"max_x": int, "max_y": int, "labels": [l],
        "size": int, "hist_c": ndarray, "hist_t": ndarray}
        labels 是一个列表. 区域, 可能是由 graphbased image segmetation 算法得到的最原始区域合并而来. 此处保留所有的 label.
        :return:
        """
        # 以图像中每一个块标签为 key, value 中存储该块的一些性质.
        R = dict()
        for y, row in enumerate(self.image_l):
            for x, (r, g, b, l) in enumerate(row):
                if l not in R:
                    R[l] = {"min_x": float('inf'), "min_y": float('inf'),
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

        for k, v in list(R.items()):
            masked_pixels = self.image[self.image_l[:, :, 3] == k]
            R[k]["size"] = len(masked_pixels)
            R[k]["hist_c"] = self._calc_colour_hist(masked_pixels)
            R[k]["hist_t"] = self._calc_texture_hist(self.lbp_image[self.image_l[:, :, 3] == k])
        return R

    @property
    def region_dict(self):
        if self._region_dict is not None:
            return self._region_dict
        else:
            self._region_dict = self._extract_regions()
        return self._region_dict

    def _generate_neighbours(self):
        # region_dict 是由 _extract_regions 函数计算出的 R.
        # 对 regions 中的每一个块标签, 分别计算其是否相交, 如果相交, 则存入 neighbours 列表, 并返回.
        def intersect(a, b):
            # 如果 a, b 的 bounding box 相交, 返回 True, 否则返回 False.
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

        R = list(self.region_dict.items())
        neighbours = []
        for current, a in enumerate(R[:-1]):
            for b in R[current + 1:]:
                if intersect(a[1], b[1]):
                    neighbours.append((a, b))
        return neighbours

    @property
    def label_neighbor(self):
        if self._label_neighbor is not None:
            return self._label_neighbor
        else:
            self._label_neighbor = self._generate_neighbours()
        return self._label_neighbor

    def _merge_regions(self, r1, r2):
        """
        在合并两个区域时, 将两个区域的性质合并成一个.
        新生成的区域的 label 标签, 会包含原始签标. 其是所有原始标签的列表.
        :param r1:
        :param r2:
        :return:
        """
        new_size = r1["size"] + r2["size"]
        rt = {
            "min_x": min(r1["min_x"], r2["min_x"]),
            "min_y": min(r1["min_y"], r2["min_y"]),
            "max_x": max(r1["max_x"], r2["max_x"]),
            "max_y": max(r1["max_y"], r2["max_y"]),
            "size": new_size,
            "hist_c": (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
            "hist_t": (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
            "labels": r1["labels"] + r2["labels"]
        }
        return rt

    def generate_selective_search_box(self):
        # region label is stored in the 4th value of each pixel [r,g,b,(region)]
        S = dict()  # 相似度字典
        for (ai, ar), (bi, br) in self.label_neighbor:
            S[(ai, bi)] = self._calc_sim(ar, br)

        while len(S) != 0:
            i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]
            t = max(self.region_dict.keys()) + 1.0
            self.region_dict[t] = self._merge_regions(self.region_dict[i], self.region_dict[j])

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
                S[(t, n)] = self._calc_sim(self.region_dict[t], self.region_dict[n])

        regions = []
        for k, r in list(self.region_dict.items()):
            regions.append({
                'rect': (
                    r['min_x'], r['min_y'],
                    r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
                'size': r['size'],
                'labels': r['labels']
            })
        return regions

    @property
    def selective_search_box(self):
        if self._selective_search_box is not None:
            return self._selective_search_box
        else:
            self._selective_search_box = self.generate_selective_search_box()
        return self._selective_search_box

    def generate_selective_search_image(self):
        """
        将 selective search box 画在 image 图上, 并返回.
        :return:
        """
        candidates = set()
        for r in self.selective_search_box:
            if r['rect'] in candidates:
                continue
            candidates.add(r['rect'])

        image_copy = self.image.copy()
        for x, y, w, h in candidates:
            cv.rectangle(image_copy, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1, lineType=cv.LINE_AA)
        return image_copy

    @property
    def selective_search_image(self):
        if self._selective_search_image is not None:
            return self._selective_search_image
        else:
            self._selective_search_image = self.generate_selective_search_image()
        return self._selective_search_image


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


if __name__ == '__main__':
    ss = SelectiveSearch(image_path="../../dataset/other/people.jpg", gbis_neighborhood_8=True, gbis_min_size=200)
    result = ss.selective_search_image
    # print(result)
    show_image(result)




