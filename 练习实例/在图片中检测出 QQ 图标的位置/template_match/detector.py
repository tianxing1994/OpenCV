import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


class TemplateMatching(object):
    def __init__(self, template_descriptors):
        self._template_descriptors = template_descriptors

        self._scene_image = None
        self._scene_keypoints = None
        self._scene_descriptors = None

        self._sift = cv.xfeatures2d.SIFT_create()
        self._flann = self._init_flann()

    def _init_scene(self, image):
        keypoints, descriptors = self._sift.detectAndCompute(image, None)
        self._scene_image = image
        self._scene_keypoints = keypoints
        self._scene_descriptors = descriptors
        return self._scene_keypoints, self._scene_descriptors

    def _init_flann(self):
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        self._flann = flann
        return self._flann

    def _knn_template_match(self, image):
        self._init_scene(image)
        # 指定 k=2, 为每一个 queryDescriptors 查找两个最佳匹配.
        matches = self._flann.knnMatch(queryDescriptors=self._template_descriptors,
                                       trainDescriptors=self._scene_descriptors,
                                       k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        k = 3
        if len(good_matches) > k:
            return good_matches
        else:
            print("Not enough matches are found - %d/%d" % (len(good_matches), k))
            return None

    def template_matching(self, image):
        good_matches = self._knn_template_match(image)
        if good_matches is not None:
            return self.calc_match_points(good_matches)
        else:
            return None

    def calc_match_points(self, good_matches):
        points = np.float32([self._scene_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        return points
