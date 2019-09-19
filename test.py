import numpy as np


population = list(range(150))
np.random.shuffle(population)
index = population[: 100]


print(index)