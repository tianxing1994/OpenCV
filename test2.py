import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


points = np.array([[[0, 0]],
                   [[0, 100]],
                   [[100, 100]],
                   [[100, 0]]], dtype=np.float)

M = np.array([[ 4.41503881e-01, -1.60450814e-01, 1.18716274e+02],
               [-4.53472783e-04, 4.08068499e-01, 1.60954123e+02],
               [-2.51163367e-04, -3.38055138e-04, 1.00000000e+00]])

dst = cv.perspectiveTransform(points, M)

print(dst)