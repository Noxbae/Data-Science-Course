import numpy as np
import matplotlib.pyplot as plt

'''
Prime numbers up to 20K
if P is a prime number, then X = P*cos(P) and Y = P*sin(P)
'''

P = [2]
prime = ("False")
for i in range (3, 20001):
    def prime_validation():
        global prime
        for l in P:
            if i % l == 0:
                prime = np.append(prime,"True")
                return
            else:
                prime = np.append(prime,"False")
    prime_validation()
    if "True" not in prime:
        P = np.append(P, i)
    prime = ("False")

plt.rcParams['figure.dpi'] = 100
plt.style.use('dark_background')

def get_coordinate(num):
    return num * np.cos(num), num * np.sin(num)

x, y = get_coordinate(P)

plt.scatter(x, y, s=1)
plt.axis("off")
plt.axis("equal")
plt.show()