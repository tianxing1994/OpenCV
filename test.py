import numpy as np


nd = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

result = np.where(nd > 3)
print(result)
print(result[0])
