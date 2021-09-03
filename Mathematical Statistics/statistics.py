import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


class RegressionAnalysis:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Fit a regression model: y = b0 + b1x1 + b2x2 + ... + bmxm
        :param X: (n_samples, n_features)
        :param y: (n_samples, )
        """
        self.LR = LinearRegression()
        self.n, self.m = X.shape
        self.X, self.y = X, y

        self.LR.fit(X, y)
        self.coef = np.concatenate(([self.LR.intercept_], self.LR.coef_))  # [b0, b1, ..., bm]
        self.MSE = ((self.LR.predict(X) - y) ** 2).sum() / (self.n - self.m - 1)  # mean square error
        self.R2 = self.LR.score(X, y)  # coefficient of determination

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.LR.predict(X)

    def f_test(self) -> float:
        F = self.R2 / (1 - self.R2) * (self.n - self.m - 1) / self.m
        return 1 - stats.f.cdf(F, self.m, self.n - self.m - 1)

    def t_test(self) -> np.ndarray:
        X = np.concatenate((np.ones((self.n, 1)), self.X), axis=1)
        T = (self.coef / np.sqrt(np.linalg.inv(X.T @ X).diagonal())) / np.sqrt(self.MSE)
        return 1 - stats.t.cdf(T, self.n - self.m - 1)
