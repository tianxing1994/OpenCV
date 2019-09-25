import random
popu = [1, 2, 3, 4, 5, 6]
weight = [1, 2, 3, 4, 5, 6]
cum_weight = [1, 2, 3, 4, 5, 6]
k = 2

x = random.choices(popu, weight, k=k)
print(x)

y = random.choices(popu, cum_weights=cum_weight, k=3)
print(y)
