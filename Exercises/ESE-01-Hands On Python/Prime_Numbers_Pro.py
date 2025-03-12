#Prime numbers up to 20K
# if P is a prime number, then X = P*cos(P) and Y = P*sin(P)

import numpy as np
import matplotlib.pyplot as plt

def es_primo(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primos_generados = [x for x in range(1, 20001) if es_primo(x)]

plt.rcParams['figure.dpi'] = 100
plt.style.use('dark_background')

def get_coordinate(num):
    return num * np.cos(num), num * np.sin(num)

x, y = get_coordinate(primos_generados)

plt.scatter(x, y, s=1)
plt.axis("off")
plt.axis("equal")
plt.show()