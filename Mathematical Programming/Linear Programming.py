"""
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html?highlight=linprog
"""

import numpy as np
from scipy.optimize import linprog


def Ex1():
    c = np.array([2, 3, 1])
    A_ub = np.array([[-1, -4, -2],
                     [-3, -2, 0]])
    b_ub = np.array([-8, -6])
    res = linprog(c=c,
                  A_ub=A_ub,
                  b_ub=b_ub,
                  bounds=[(0, None), (0, None), (0, None)],
                  method='revised simplex')
    print(res)


def Ex2():
    c = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    A_ub = np.array([[1, -1, -1, 1, -1, 1, 1, -1],
                     [1, -1, 1, -3, -1, 1, -1, 3],
                     [1, -1, -2, 3, -1, 1, 2, -3]])
    b_ub = np.array([-2, -1, -1/2])
    res = linprog(c=c,
                  A_ub=A_ub,
                  b_ub=b_ub,
                  bounds=[(0, None)] * 8,
                  method='revised simplex')
    print(res)


Ex1()
