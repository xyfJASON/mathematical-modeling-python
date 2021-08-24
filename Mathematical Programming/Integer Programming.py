import numpy as np
from typing import Optional
from scipy.optimize import linprog
from scipy.optimize import linear_sum_assignment


class IntegerProgramming:
    def __init__(self,
                 c: np.ndarray,
                 A_ub: Optional[np.ndarray] = None,
                 b_ub: Optional[np.ndarray] = None,
                 A_eq: Optional[np.ndarray] = None,
                 b_eq: Optional[np.ndarray] = None) -> None:
        self.c = c
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq

    def branch_bound(self,
                     method: str = 'revised simplex',
                     bounds: Optional[list] = None,
                     is_int: Optional[np.ndarray] = None):

        def check_int(result) -> bool:
            return np.equal(np.floor(result.x[is_int]), result.x[is_int]).all()

        def get_not_int(result) -> list:
            return [i for i in range(self.c.shape[0]) if is_int[i] and np.floor(result.x[i]) != result.x[i]]

        q = [bounds]  # queue
        best = None
        while len(q):
            cur = q.pop(0)
            res = linprog(self.c, self.A_ub, self.b_ub, self.A_eq, self.b_eq, cur, method)
            if not res.success:
                continue
            if check_int(res):
                if best is None or res.fun < best.fun:
                    best = res
            else:
                if res.fun >= best.fun:
                    continue
                idxs = get_not_int(res)
                for idx in idxs:
                    now = cur.copy()
                    now[idx][1] = np.floor(res.x[idx])
                    q.append(now)
                    now = cur.copy()
                    now[idx][0] = np.ceil(res.x[idx])
                    q.append(now)
        return best

    def implicit_enumerate(self,
                           known_solution: Optional[np.ndarray] = None,
                           verbose: bool = False) -> tuple[np.ndarray, float]:
        c = self.c.copy()
        A_ub = self.A_ub.copy() if self.A_ub is not None else None
        b_ub = self.b_ub.copy() if self.b_ub is not None else None
        A_eq = self.A_eq.copy() if self.A_eq is not None else None
        b_eq = self.b_eq.copy() if self.b_eq is not None else None
        argneg = np.argwhere(c < 0).flatten()
        c[argneg] = -c[argneg]
        if A_ub is not None:
            b_ub -= A_ub[:, argneg].sum(axis=1)
            A_ub[:, argneg] = -A_ub[:, argneg]
        if A_eq is not None:
            b_eq -= A_eq[:, argneg].sum(axis=1)
            A_eq[:, argneg] = -A_eq[:, argneg]
        argidx = np.argsort(c)
        iargidx = np.argsort(argidx)
        bestans, bestx = c.sum(), np.zeros(len(c))
        if known_solution is not None:
            tmp_solution = known_solution.copy()
            tmp_solution[argneg] = 1 - tmp_solution[argneg]
            bestans = c.T @ tmp_solution
            bestx = tmp_solution[argidx]

        def dfs(i, nowans, nowx) -> None:
            nonlocal bestans, bestx
            if verbose:
                print(nowx, nowans, bestans)
            if i == len(self.c):
                ok = True
                if A_eq is not None:
                    ok &= np.all((A_eq @ np.array(nowx)[iargidx]) == b_eq)
                if A_ub is not None:
                    ok &= np.all((A_ub @ np.array(nowx)[iargidx]) <= b_ub)
                if ok:
                    bestans, bestx = nowans.copy(), nowx.copy()
                return
            for t in [0, 1]:
                if nowans + t * c[argidx[i]] < bestans:
                    nowx.append(t)
                    dfs(i+1, nowans + t * c[argidx[i]], nowx)
                    nowx.pop()

        dfs(0, 0.0, [])
        bestx = np.array(bestx)[iargidx]
        bestx[argneg] = 1 - bestx[argneg]
        return bestx, self.c.T @ bestx


def Ex1():
    c = np.array([[3, 8, 2, 10, 3],
                  [8, 7, 2, 9, 7],
                  [6, 4, 2, 7, 5],
                  [8, 4, 2, 3, 5],
                  [9, 10, 6, 9, 10]])
    A_eq = np.vstack((
        np.concatenate(([1] * 5, [0] * 20)),
        np.concatenate(([0] * 5, [1] * 5, [0] * 15)),
        np.concatenate(([0] * 10, [1] * 5, [0] * 10)),
        np.concatenate(([0] * 15, [1] * 5, [0] * 5)),
        np.concatenate(([0] * 20, [1] * 5)),
        np.array([1, 0, 0, 0, 0] * 5),
        np.array([0, 1, 0, 0, 0] * 5),
        np.array([0, 0, 1, 0, 0] * 5),
        np.array([0, 0, 0, 1, 0] * 5),
        np.array([0, 0, 0, 0, 1] * 5)
    ))
    b_eq = np.array([1] * 10)
    bounds = [(0, 1)] * 25
    is_int = np.array([True] * 25)

    row_id, col_id = linear_sum_assignment(c)
    print(c[row_id, col_id].sum())  # 21
    print()

    solver = IntegerProgramming(c.flatten(), A_eq=A_eq, b_eq=b_eq)
    res = solver.branch_bound(bounds=bounds, is_int=is_int)
    print(res)  # 21
    print()

    res = solver.implicit_enumerate(known_solution=np.diag(np.ones(5)).flatten())
    print(res)  # 21


def Ex2():
    c = np.array([-3, -2, -1])
    A_ub = np.array([[1, 1, 1]])
    b_ub = np.array([7])
    A_eq = np.array([[4, 2, 1]])
    b_eq = np.array([12])
    bounds = [(0, None), (0, None), (0, 1)]
    is_int = np.array([False, False, True])

    solver = IntegerProgramming(c, A_ub, b_ub, A_eq, b_eq)
    res = solver.branch_bound(bounds=bounds, is_int=is_int)
    print(res)  # -12


Ex1()
