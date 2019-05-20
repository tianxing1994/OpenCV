import cv2 as cv
import numpy as np
import argparse
import shelve
import imagehash
import glob
from PIL import Image


image_wechat = Image.open('dataset/other/wechat.jpg')
image_wechat_result = Image.open('dataset/other/wechat_result.jpg')

h_image_wechat = str(imagehash.dhash(image_wechat))
h_image_wechat_result = str(imagehash.dhash(image_wechat_result))

# result_dhash = imagehash.hex_to_hash(h)

hsh = cv.img_hash.BlockMeanHash_create()
cv_image_wechat = hsh.compute(np.array(image_wechat, dtype=np.uint8))
cv_image_wechat_result = hsh.compute(np.array(image_wechat, dtype=np.uint8))

print(h_image_wechat)
print(h_image_wechat_result)
print(cv_image_wechat)
print(cv_image_wechat_result)

print(imagehash.hex_to_hash(h_image_wechat) - imagehash.hex_to_hash(h_image_wechat_result))

# 两张图是有差别的,  opencv 检测出来的结果是完全相同啊.
print(cv_image_wechat == cv_image_wechat_result)