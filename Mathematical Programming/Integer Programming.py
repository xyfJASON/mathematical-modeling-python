import numpy as np
from typing import Optional
from scipy.optimize import linprog
from scipy.optimize import linear_sum_assignment


def IP_branch_bound(c: np.ndarray,
                    A_ub: Optional[np.ndarray] = None,
                    b_ub: Optional[np.ndarray] = None,
                    A_eq: Optional[np.ndarray] = None,
                    b_eq: Optional[np.ndarray] = None,
                    bounds: Optional[list] = None,
                    method: Optional[str] = 'revised simplex',
                    is_int: Optional[np.ndarray] = None):

    def check_int(result) -> bool:
        return np.equal(np.floor(result.x[is_int]), result.x[is_int]).all()

    def get_not_int(result) -> list:
        return [i for i in range(c.shape[0]) if is_int[i] and np.floor(result.x[i]) != result.x[i]]

    q = [bounds]  # queue
    best = None
    while len(q):
        cur = q.pop(0)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, cur, method)
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

    res = IP_branch_bound(c.flatten(),
                          A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds,
                          is_int=is_int)
    print(res)  # 21

    row_id, col_id = linear_sum_assignment(c)
    print(c[row_id, col_id].sum())  # 21


def Ex2():
    c = np.array([-3, -2, -1])
    A_ub = np.array([[1, 1, 1]])
    b_ub = np.array([7])
    A_eq = np.array([[4, 2, 1]])
    b_eq = np.array([12])
    bounds = [(0, None), (0, None), (0, 1)]
    is_int = np.array([False, False, True])

    res = IP_branch_bound(c, A_ub, b_ub, A_eq, b_eq,
                          bounds=bounds,
                          is_int=is_int)
    print(res)  # -12


Ex1()
