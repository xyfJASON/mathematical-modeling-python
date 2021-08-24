import numpy as np


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


def Ex():
    CB1 = np.array([[1, 1/4, 1/2], [4, 1, 3], [2, 1/3, 1]])
    CB2 = np.array([[1, 1/4, 1/5], [4, 1, 1/2], [5, 2, 1]])
    CB3 = np.array([[1, 3, 1/3], [1/3, 1, 1/7], [3, 7, 1]])
    CB4 = np.array([[1, 1/3, 5], [3, 1, 7], [1/5, 1/7, 1]])
    CB5 = np.array([[1, 1, 7], [1, 1, 7], [1/7, 1/7, 1]])
    CB6 = np.array([[1, 7, 9], [1/7, 1, 1], [1/9, 1, 1]])
    BA = np.array([[1, 1, 1, 4, 1, 1/2],
                   [1, 1, 2, 4, 1, 1/2],
                   [1, 1/2, 1, 5, 3, 1/2],
                   [1/4, 1/4, 1/5, 1, 1/3, 1/3],
                   [1, 1, 1/3, 3, 1, 1],
                   [2, 2, 2, 3, 3, 1]])
    judge_mat = [[CB1, CB2, CB3, CB4, CB5, CB6], [BA]]

    solver = AHP([3, 6, 1], judge_mat)
    res = solver.run()
    print(res)


Ex()
