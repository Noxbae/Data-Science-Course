#Prime numbers up to 20K
# if P is a prime number, then X = P*cos(P) and Y = P*sin(P)

import numpy as np
import matplotlib.pyplot as plt
import sympy

primes = sympy.primerange(1,20001)
nums = np.array(list(primes))

plt.rcParams['figure.dpi'] = 100
plt.style.use('dark_background')

def get_coordinate(num):
    return num * np.cos(num), num * np.sin(num)

x, y = get_coordinate(nums)

plt.scatter(x, y, s=1)
plt.axis("off")
plt.axis("equal")
plt.show()