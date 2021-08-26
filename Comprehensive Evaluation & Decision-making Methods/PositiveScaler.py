import numpy as np
from typing import Optional


def positive_scale(x: np.ndarray,
                   kind: np.ndarray,
                   mid_best: Optional[np.ndarray] = None,
                   interval_best: Optional[np.ndarray] = None) -> np.ndarray:
    """ 数据正向化
    :param x: (N, *), N is the number of data
    :param kind: (*)
                 0: 极大型指标
                 1: 极小型指标
                 2: 中间型指标
                 3: 区间型指标
    :param mid_best: (*), 中间型指标的最佳值
    :param interval_best: (*, 2), 区间型指标的最佳区间，该区间内值化为 1
    """
    assert x.shape[1:] == kind.shape
    if mid_best is not None:
        assert kind.shape == mid_best.shape
    if interval_best is not None:
        assert kind.shape == interval_best.shape[:-1]
        assert interval_best.shape[-1] == 2
    mask0, mask1, mask2, mask3 = kind == 0, kind == 1, kind == 2, kind == 3
    y = x.copy()
    if mask0.any():
        y[:, mask0] = (y[:, mask0] - y[:, mask0].min(0)) / (y[:, mask0].max(0) - y[:, mask0].min(0) + 1e-12)
    if mask1.any():
        y[:, mask1] = (y[:, mask1].max(0) - y[:, mask1]) / (y[:, mask1].max(0) - y[:, mask1].min(0) + 1e-12)
    if mask2.any():
        y[:, mask2] = np.fabs(y[:, mask2] - mid_best[mask2])
        y[:, mask2] = 1 - y[:, mask2] / (y[:, mask2].max(0) + 1e-12)
    if mask3.any():
        interval_best = interval_best[mask3]
        tmp = y[:, mask3]
        ml = tmp < interval_best[:, 0]
        mr = tmp > interval_best[:, 1]
        m1 = ~(ml | mr)
        M = np.maximum(interval_best[:, 0] - tmp.min(0),
                       tmp.max(0) - interval_best[:, 1])
        tmp[m1] = 1
        tmp[ml] = 1 - (interval_best[:, 0] - y[:, mask3][ml]) / (M + 1e-12)
        tmp[mr] = 1 - (y[:, mask3][mr] - interval_best[:, 1]) / (M + 1e-12)
        y[:, mask3] = tmp
    return y


def main():
    x = np.array([[4.69, 6.59, 51, 11.94],
                  [2.03, 7.86, 19, 6.46],
                  [9.11, 6.31, 46, 8.91],
                  [8.61, 7.05, 46, 26.43],
                  [7.13, 6.5, 50, 23.57],
                  [2.39, 6.77, 38, 24.62],
                  [7.69, 6.79, 38, 6.01],
                  [9.3, 6.81, 27, 31.57],
                  [5.45, 7.62, 5, 18.46],
                  [6.19, 7.27, 17, 7.51],
                  [7.93, 7.53, 9, 6.52],
                  [4.4, 7.28, 17, 25.3]])
    kind = np.array([0, 2, 1, 3])
    mid_best = np.array([0, 7, 0, 0])
    interval_best = np.zeros((4, 2))
    interval_best[3, :] = np.array([10, 20])

    y = positive_scale(x=x, kind=kind, mid_best=mid_best, interval_best=interval_best)
    np.set_printoptions(suppress=True, precision=3)
    print(y)


if __name__ == '__main__':
    main()
