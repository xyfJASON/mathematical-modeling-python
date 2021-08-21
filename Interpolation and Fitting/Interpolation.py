import numpy as np
from scipy.interpolate import *
import matplotlib.pyplot as plt


def Ex1():
    x = np.arange(0, 10)
    y = np.sin(x * 3)

    Zero = interp1d(x, y, kind='zero')
    Linear = interp1d(x, y, kind='linear')
    Quadratic = interp1d(x, y, kind='quadratic')
    Cubic = interp1d(x, y, kind='cubic')
    Lagrange = lagrange(x, y)

    x_pred = np.arange(1.5, 7.5, 0.01)
    y_true = np.sin(x_pred * 3)
    y_zero = Zero(x_pred)
    y_linear = Linear(x_pred)
    y_quadratic = Quadratic(x_pred)
    y_cubic = Cubic(x_pred)
    y_lagrange = Lagrange(x_pred)

    fig, ax = plt.subplots(1)
    ax.plot(x, y, 'o')
    ax.plot(x_pred, y_true, label='true')
    ax.plot(x_pred, y_zero, label='zero')
    ax.plot(x_pred, y_linear, label='linear')
    ax.plot(x_pred, y_quadratic, label='quadratic')
    ax.plot(x_pred, y_cubic, label='cubic')
    ax.plot(x_pred, y_lagrange, label='lagrange')
    plt.legend()
    plt.show()


def Ex2():
    x = np.array([100, 200, 300, 400, 500])
    y = np.array([100, 200, 300, 400])
    z = np.array([[636, 698, 680, 662],
                  [697, 712, 674, 626],
                  [624, 630, 598, 552],
                  [478, 478, 412, 334],
                  [450, 420, 400, 310]])
    rect = RectBivariateSpline(x, y, z, kx=3, ky=3)

    x_pred = np.arange(100, 510, 10)
    y_pred = np.arange(100, 510, 10)
    z_rect = rect(x_pred, y_pred)
    pos = np.unravel_index(z_rect.argmax(), z_rect.shape)
    print('max position: %d %d' % (x_pred[pos[0]], y_pred[pos[1]]))
    print('max value: %.4f' % z_rect[pos[0], pos[1]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax.scatter3D(X.flatten(), Y.flatten(), z.transpose().flatten())
    X, Y = np.meshgrid(x_pred, y_pred)
    ax.plot_surface(X, Y, z_rect.transpose(), cmap='viridis', alpha=0.95)
    plt.show()


def Ex3():
    x = np.array([129, 140, 103.5, 88, 185.5, 195, 105, 157.5, 107.5, 77, 81, 162, 162, 117.5])
    y = np.array([7.5, 141.5, 23, 147, 22.5, 137.5, 85.5, -6.5, -81, 3, 56.5, -66.5, 84, -33.5])
    z = -np.array([4, 8, 6, 8, 6, 8, 8, 9, 9, 8, 8, 9, 4, 9])
    x_pred = np.linspace(np.min(x), np.max(x), 100)
    y_pred = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(x_pred, y_pred)
    z_cubic = griddata(np.vstack((x, y)).transpose(), z,
                       xi=np.vstack((X.flatten(), Y.flatten())).transpose(),
                       method='cubic')
    z_nearest = griddata(np.vstack((x, y)).transpose(), z,
                         xi=np.vstack((X.flatten(), Y.flatten())).transpose(),
                         method='nearest')
    z_cubic[np.isnan(z_cubic)] = z_nearest[np.isnan(z_cubic)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z)
    ax.plot_surface(X, Y, z_cubic.reshape(100, 100), cmap='viridis')
    plt.show()


Ex3()
