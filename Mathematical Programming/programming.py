import numpy as np
from typing import Optional
from scipy.optimize import linprog


def branch_bound(c: np.ndarray,
                 A_ub: Optional[np.ndarray] = None,
                 b_ub: Optional[np.ndarray] = None,
                 A_eq: Optional[np.ndarray] = None,
                 b_eq: Optional[np.ndarray] = None,
                 method: str = 'revised simplex',
                 bounds: Optional[list] = None,
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


def implicit_enumerate(c: np.ndarray,
                       A_ub: Optional[np.ndarray] = None,
                       b_ub: Optional[np.ndarray] = None,
                       A_eq: Optional[np.ndarray] = None,
                       b_eq: Optional[np.ndarray] = None,
                       known_solution: Optional[np.ndarray] = None,
                       verbose: bool = False) -> tuple[np.ndarray, float]:
    _c = c.copy()
    _A_ub = A_ub.copy() if A_ub is not None else None
    _b_ub = b_ub.copy() if b_ub is not None else None
    _A_eq = A_eq.copy() if A_eq is not None else None
    _b_eq = b_eq.copy() if b_eq is not None else None
    argneg = np.argwhere(_c < 0).flatten()
    _c[argneg] = -_c[argneg]
    if _A_ub is not None:
        _b_ub -= _A_ub[:, argneg].sum(axis=1)
        _A_ub[:, argneg] = -_A_ub[:, argneg]
    if _A_eq is not None:
        _b_eq -= _A_eq[:, argneg].sum(axis=1)
        _A_eq[:, argneg] = -_A_eq[:, argneg]
    argidx = np.argsort(_c)
    iargidx = np.argsort(argidx)
    bestans, bestx = _c.sum(), np.zeros(len(_c))
    if known_solution is not None:
        tmp_solution = known_solution.copy()
        tmp_solution[argneg] = 1 - tmp_solution[argneg]
        bestans = _c.T @ tmp_solution
        bestx = tmp_solution[argidx]

    def dfs(i, nowans, nowx) -> None:
        nonlocal bestans, bestx
        if verbose:
            print(nowx, nowans, bestans)
        if i == len(c):
            ok = True
            if _A_eq is not None:
                ok &= np.all((_A_eq @ np.array(nowx)[iargidx]) == _b_eq)
            if _A_ub is not None:
                ok &= np.all((_A_ub @ np.array(nowx)[iargidx]) <= _b_ub)
            if ok:
                bestans, bestx = nowans.copy(), nowx.copy()
            return
        for t in [0, 1]:
            if nowans + t * _c[argidx[i]] < bestans:
                nowx.append(t)
                dfs(i+1, nowans + t * _c[argidx[i]], nowx)
                nowx.pop()

    dfs(0, 0.0, [])
    bestx = np.array(bestx)[iargidx]
    bestx[argneg] = 1 - bestx[argneg]
    return bestx, c.T @ bestx
