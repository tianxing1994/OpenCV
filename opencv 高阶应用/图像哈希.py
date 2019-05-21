import cv2 as cv
import numpy as np
import argparse
import shelve
import imagehash
import glob
from PIL import Image


def base_demo():
    image_wechat = Image.open('../dataset/other/wechat.jpg')
    image_wechat_result = Image.open('../dataset/other/wechat_result.jpg')

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
    return

def index():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of images")
	ap.add_argument("-s", "--shelve", required=True, help="output shelve database")
	args = vars(ap.parse_args())

	# open the shelve database
	db = shelve.open(args["shelve"], writeback=True)

	# loop over the image dataset
	for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
		# load the image and compute the difference hash
		image = Image.open(imagePath)
		h = str(imagehash.dhash(image))

		# extract the filename from the path and update the database
		# using the hash as the key and the filename append to the
		# list of values
		filename = imagePath[imagePath.rfind("/") + 1:]
		db[h] = db.get(h, []) + [filename]
	print(dict(db))
	# close the shelf database
	db.close()
	return


def search():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to dataset of images")
    ap.add_argument("-s", "--shelve", required=True, help="output shelve database")
    ap.add_argument("-q", "--query", required=True, help="path to the query image")
    args = vars(ap.parse_args())

    # open the shelve database
    db = shelve.open(args["shelve"])

    # load the query image, compute the difference image hash, and
    # and grab the images from the database that have the same hash
    # value
    query = Image.open(args["query"])
    h = str(imagehash.dhash(query))
    filenames = db[h]
    print("Found %d images" % (len(filenames)))

    # loop over the images
    for filename in filenames:
        # image = Image.open(args["dataset"] + "/" + filename)
        image = Image.open(filename)
        print(filename, image)
        image.show()

    # close the shelve database
    db.close()
    return


if __name__ == '__main__':
    base_demo()