import tensorflow as tf
import cv2 as cv
import numpy as np
from collections import  Counter


l = [1, 2, 3, 1, 2, 3, 4, 5, 1]

def get_majority_class_count(y_train):
    counter_dict = dict(Counter(y_train))
    counter_sorted = sorted(counter_dict.items(), key=lambda x: x[1], reverse=True)
    result = counter_sorted[0][0]

    return result

result = get_majority_class_count(l)

print(result)