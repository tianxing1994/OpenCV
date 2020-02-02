import os
import cv2 as cv
import numpy as np

from template_match.detector import TemplateMatching
from template_match.load_data import get_sample_by_label_list
from template_match.template_descriptors import DescriptorsManager
from template_match.config import PROJECT_PATH


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


if __name__ == '__main__':
    dataset = get_sample_by_label_list(cls_list=[1, 3, 4, 5], channel_list=[0, 1])
    descriptors_manager = DescriptorsManager(dataset=dataset, n_clusters=300)
    template_descriptors = descriptors_manager.kmeans_descriptors
    tm = TemplateMatching(template_descriptors=template_descriptors)

    image_name_list = os.listdir(os.path.join(PROJECT_PATH, 'template_match/dataset/image'))

    for image_name in image_name_list:
        image_path = os.path.join(PROJECT_PATH, 'template_match/dataset/image', image_name)
        image = cv.imread(image_path)

        result = tm.template_matching(image)
        if result is None:
            continue
        for point in result:
            center = tuple(point.astype(np.int))
            cv.circle(image, center=center, radius=2, color=(0, 0, 255), thickness=2, lineType=cv.LINE_8)

        show_image(image)
