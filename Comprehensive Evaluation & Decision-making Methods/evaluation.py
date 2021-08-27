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


class TOPSIS:
    def __init__(self,
                 x: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> None:
        assert len(x.shape) == 2 and (weights is None or len(weights.shape) == 1)
        assert (x >= 0).all() and (x <= 1).all()
        assert weights is None or weights.shape[0] == x.shape[1]
        self.x = x
        self.weights = np.ones(x.shape[1]) if weights is None else weights

    def run(self) -> np.ndarray:
        b = self.x / np.sqrt(np.sum(self.x ** 2, axis=0))
        c = self.weights * b
        c_pos = c.max(axis=0)
        c_neg = c.min(axis=0)
        d_pos = np.sqrt(np.sum((c - c_pos) ** 2, axis=1))
        d_neg = np.sqrt(np.sum((c - c_neg) ** 2, axis=1))
        f = d_neg / (d_neg + d_pos)
        return f


class GreyRelationalAnalysis:
    def __init__(self,
                 x: np.ndarray,
                 x0: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None,
                 rho: float = 0.5) -> None:
        """
        :param x: (m, n), m is the number of evaluation objects,
                          n is the number of evaluation indicators
        :param x0: reference sequence or evaluation criterion
                   default is the combination of best value in the data of each indicator
        :param weights: weights of each indicator, default is uniform
        :param rho: coefficient in [0, 1], default is 0.5
        """
        assert len(x.shape) == 2
        m, n = x.shape
        if x0 is not None:
            assert x0.shape == (n, )
        if weights is not None:
            assert weights.shape == (n, )
        self.x = x
        self.x0 = np.max(x, axis=0) if x0 is None else x0
        self.weights = np.ones(n) / n if weights is None else weights
        self.rho = rho

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        tmp = np.fabs(self.x - self.x0)
        minmin = np.min(tmp)
        maxmax = np.max(tmp)
        R = (minmin + self.rho * maxmax) / (tmp + self.rho * maxmax)
        return R, R @ self.weights


class AHP:
    def __init__(self,
                 sz_layers: list[int],
                 judge_mat: list[list[np.ndarray]]) -> None:
        """
        :param sz_layers: number of elements in each layer
        :param judge_mat: judgement matrices
        """
        self.n_layers = len(sz_layers)
        assert len(judge_mat) == self.n_layers - 1
        for i in range(self.n_layers - 1):
            assert np.stack(judge_mat[i], axis=0).shape == (sz_layers[i+1], sz_layers[i], sz_layers[i])
        for sz in sz_layers:
            assert 1 <= sz <= 10
        self.sz_layers = sz_layers
        self.judge_mat = judge_mat

        self.CI = []
        self.RI = []
        self.RI_table = [None, 0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
        self.b = []
        self.a = []

    def run(self) -> np.ndarray:
        for i in range(self.n_layers - 1):
            b, CI, RI = [], [], []
            for j in range(self.sz_layers[i+1]):
                mat = self.judge_mat[i][j]
                eigw, eigv = np.linalg.eig(mat)
                eigv, eigw = eigv[:, np.argmax(eigw)], np.max(eigw)
                eigv, eigw = eigv.real, eigw.real
                CI.append((eigw - self.sz_layers[i]) / (self.sz_layers[i] - 1))
                RI.append(self.RI_table[self.sz_layers[i]])
                assert CI[-1] / RI[-1] < 0.10, 'Cannot pass consistency test.'
                b.append(eigv / eigv.sum())
            self.b.append(b)
            self.CI.append(CI)
            self.RI.append(RI)
        self.a.append(np.array([1.0]))
        for i in range(self.n_layers - 2, -1, -1):
            a = []
            for j in range(self.sz_layers[i]):
                a.append(self.a[-1] @ np.vstack(self.b[i])[:, j])
            CR = (self.a[-1] @ np.array(self.CI[i])) / (self.a[-1] @ np.array(self.RI[i]))
            assert CR < 0.10, 'Cannot pass consistency test.'
            self.a.append(np.array(a))
        return self.a[-1]


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
