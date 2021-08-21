import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def plot_curve(x, y):
    assert x.shape[0] == y.shape[1]
    fig, ax = plt.subplots(1, 1)
    for i, yi in enumerate(y):
        ax.plot(x, yi, label=f'y{i}')
    plt.legend()
    plt.show()


def Ex1():
    def f(x, y):
        return -2 * y + 2 * x * x + 2 * x

    x_range = (0, 0.5)
    y0 = np.array([1])

    res = solve_ivp(fun=f,
                    t_span=x_range,
                    y0=y0,
                    method='RK45',
                    t_eval=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    print(res)

    t_eval = np.linspace(0, 0.5, 100)
    res = solve_ivp(f, x_range, y0, t_eval=t_eval)
    plot_curve(t_eval, res.y)


def Ex2():
    def f(_, y):
        return np.array([y[1], y[2], 3*y[2]+y[0]*y[1]])

    x_range = (0, 2)
    y0 = np.array([0, 1, -1])

    res = solve_ivp(fun=f,
                    t_span=x_range,
                    y0=y0,
                    method='RK45',
                    t_eval=[0, 0.5, 1])
    print(res)

    t_eval = np.linspace(0, 2, 100)
    res = solve_ivp(f, x_range, y0, t_eval=t_eval)
    plot_curve(t_eval, res.y)


Ex2()
