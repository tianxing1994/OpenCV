import heapq
from math import log

import math


def shortestPath(G, start, end, length_penalty=0.0):
    def flatten(L):
        while len(L) > 0:
            yield L[0]
            L = L[1]
        return

    q = [(0, start, ())]
    visited = set()
    while True:
        (cost, v1, path) = heapq.heappop(q)
        if v1 not in visited:
            visited.add(v1)
            if v1 == end:
                return list(flatten(path))[::-1] + [v1]
            path = (v1, path)
            for (v2, cost2) in G[v1].iteritems():
                if v2 not in visited:
                    heapq.heappush(q, (cost + cost2 + length_penalty * log(len(visited)), v2, path))


class LiveWireSegmentation(object):
    def __init__(self, image=None, smooth_image=False, threshold_gradient_image=False):
        super(LiveWireSegmentation, self).__init__()
        self._image = None
        self.edges = None
        self.G = None
        self.smoot_image = smooth_image
        self.threshold_gradient_image = threshold_gradient_image
        self.image = image

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value

        if self._image is not None:
            if self.smoot_image:
                self._smooth_image()

            self._compute_gradient_image()

            if self.threshold_gradient_image:
                self._threshold_gradient_image()

            self._compute_graph()
        else:
            self.edges = None
            self.G = None

    def _smooth_image(self):
        from skimage import restoration
        self._image = restoration.denoise_bilateral(self.image)

    def _compute_gradient_image(self):
        from skimage import filters
        self.edges = filters.scharr(self._image)

    def _threshold_gradient_image(self):
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(self.edges)
        self.edges = self.edges > threshold
        self.edges = self.edges.astype(float)

    def _compute_graph(self, norm_function=math.fabs):
        self.G = {}
        rows, cols = self.edges.shape
        for col in range(cols):
            for row in range(rows):
                neighbors = []
                if row > 0:
                    neighbors.append((row-1, col))
                if row < rows-1:
                    neighbors.append((row+1, col))
                if col > 0:
                    neighbors.append((row, col-1))
                if col < cols-1:
                    neighbors.append((row, col+1))

                dist = {}
                for n in neighbors:
                    dist[n] = norm_function(self.edges[row][col] - self.edges[n[0], n[1]])

                self.G[(row, col)] = dist

    def compute_shortest_path(self, from_, to_, length_penalty=0.0):
        if self.image is None:
            raise AttributeError("Load an image first!")
        path = shortestPath(self.G, from_, to_, length_penalty=length_penalty)
        return path
