import numpy as np
from scipy.optimize import lsq_linear, curve_fit


def Ex1():
    x = np.array([1990, 1991, 1992, 1993, 1994, 1995, 1996])
    y = np.array([70, 122, 144, 152, 174, 196, 202])
    A = np.stack((np.ones(x.shape[0]), x), axis=1)
    b = y

    res = np.linalg.inv(A.T @ A) @ A.T @ y
    print('#1:', res)

    res = lsq_linear(A, b)
    print('#2:', res)

    res = curve_fit(lambda x, a0, a1: a0 + a1 * x, x, y)
    print('#3:', res)


def Ex2():
    x = np.arange(1, 9)
    y = np.array([15.3, 20.5, 27.4, 36.6, 49.1, 65.6, 87.87, 117.6])

    res = curve_fit(lambda x, a, b: a * np.exp(b * x), x, y)
    print(res)

    res = curve_fit(lambda x, a, b: np.log(a) + b * x, x, np.log(y))
    print(res)


Ex2()
