from skimage import data, segmentation, color
from skimage.future import graph
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt




image = data.load('brick.png')
lbp = local_binary_pattern(image, 24, 3, "var")

print(lbp.dtype)
print(lbp.max())
print(lbp[200][200])
print(lbp.shape)



