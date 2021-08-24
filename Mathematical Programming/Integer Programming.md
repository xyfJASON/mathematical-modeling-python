<h1 style="text-align:center"> 整数规划 </h1>
<h2 style="text-align:center"> Integer Programming </h2>
<div style="text-align:center"> xyfJASON </div>



## 1 引入 `0-1` 变量的场景



### 1.1 多选一约束

例如有如下两个约束：
$$
5x_1+4x_2\leqslant 24\quad\text{or}\quad 7x_1+3x_2\leqslant 45
$$
满足其中一个约束即可。为了统一起来，引入 0-1 变量：
$$
y=\begin{cases}0&\text{选第一个约束}\\1&\text{选第二个约束}\end{cases}
$$
那么约束条件可改写为：
$$
\begin{cases}5x_1+4x_2\leqslant 24+yM\\7x_1+3x_2\leqslant 45+(1-y)M\end{cases}
$$
其中 $M$​ 是充分大的正常数。



### 1.2 多选多约束

例如有如下三个约束，要求满足至少两个约束：
$$
2x+3y\leqslant 100,\quad x+y\leqslant 50,\quad x+2y\leqslant 80
$$
那么引入三个 0-1 变量 $z_1,z_2,z_3$ 分别表示选第 $1,2,3$ 个约束，则约束条件可改写为：
$$
\begin{cases}
2x+3y\leqslant 100+(1-z_1)M\\
x+y\leqslant 50+(1-z_2)M\\
x+2y\leqslant 80+(1-z_3)M\\
z_1+z_2+z_3=2
\end{cases}
$$
 其中 $M$ 是充分大的正常数。



### 1.3 分段线性函数

例如目标函数是：
$$
z=\begin{cases}
3+4x&0\leqslant x< 2\\
15-2x&2\leqslant x<3\\
6+x&3\leqslant x<7
\end{cases}
$$
其通用的一种建模技巧如下：

设 $n$ 段线性函数 $f(x)$ 分点为 $b_1<b_2<\cdots<b_{n+1}$，引入变量 $w_k$ 和 0-1 变量 $z_k$ 满足：
$$
\begin{cases}
w_1\leqslant z_1,\,w_2\leqslant z_1+z_2,\,\cdots,\,w_n\leqslant z_{n-1}+z_n,\,w_{n+1}\leqslant z_n\\
z_1+z_2+\cdots+z_n=1,\quad z_k\in\{0,1\}\\
w_1+w_2+\cdots+w_{n+1}=1,\quad w_k\geqslant 0
\end{cases}
$$
那么 $x$ 和 $f(x)$ 就可以表示如下：
$$
\begin{align}
x&=\sum_{k=1}^{n+1}w_kb_k\\
f(x)&=\sum_{k=1}^{n+1}w_kf(b_k)
\end{align}
$$


### 1.4 固定费用问题

设目标函数中存在这样的项：
$$
P=\begin{cases}k+cx&x>0\\0&x=0\end{cases}
$$
那么引入 0-1 变量 $y=[x>0]$，则添加如下条件：
$$
y\varepsilon\leqslant x\leqslant yM
$$
其中 $\varepsilon$ 为充分小正常数，$M$ 为充分大正常数。现在目标函数中 $P$ 这一项只写作 $k+cx$ 即可。



---



## 2 算法

**非线性整数规划**没有通用的算法，事实上**非线性规划**都没有通用解法。

下列方法中，蒙特卡洛法可用于求解非线性整数规划，其余算法仅针对**线性整数规划**。

本文默认最小化目标函数，不等式约束条件均为小于等于。



### 2.1 分支定界法——整数规划（纯/混合）

忽略整数条件，单纯形法得到解作为原问题的上界。

如果当前解恰是整数解，则问题解决；否则，选取一个不满足整数条件的变量 $x_i$，设当前解中它的值为小数 $x_i^*$，则分别添加约束条件 $x\leqslant \lfloor x_i^*\rfloor$ 和 $x\geqslant \lceil x_i^*\rceil$ 得到两个新问题，分别单纯形法求解（分支）。

对于一个新问题而言，如果它是整数解，则更新原问题的下界；如果它不是整数解，若它比当前下界更优，则更新原问题的上界（定界）；否则裁剪该分支（剪枝）。然后选取一个未被裁剪的分支，重复上述步骤，直至上下界相等。



### 2.2 隐枚举法——0-1整数规划

对暴力枚举的优化，在得到一个可行解之后，容易知道更优的解必须小于该可行解，这提供了一个强力的剪枝。



### 2.3 匈牙利算法——指派问题（0-1整数规划特例）

指派问题，可以看作 0-1 规划问题求解，也可以看作二分图匹配问题，使用匈牙利算法求解。

不过 `scipy` 已经提供了指派问题的求解函数：`scipy.optimize.linear_sum_assignment`，所以我们不必实现。

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html



### 2.4 蒙特卡洛法——非线性整数规划

蒙特卡洛法就是随机取样法，随机从可行域中取点、代入、更新答案，取点充分多后能够接近最优解。



---



## 3 代码模板

以下是分支定界法和隐枚举法的算法模板：

```python
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
```



---



## 4 例题



### 4.1 例一

求解指派问题，已知指派矩阵如下：
$$
\begin{bmatrix}
3&8&2&10&3\\8&7&2&9&7\\6&4&2&7&5\\8&4&2&3&5\\9&10&6&9&10
\end{bmatrix}
$$
指派问题自然可以直接用 `linear_sum_assignment` 求解，但也可以化为如下整数规划问题：
$$
\min\quad
\begin{align}
&3x_{11}+8x_{12}+2x_{13}+10x_{14}+3x_{15}\\
+&8x_{21}+7x_{22}+2x_{23}+9x_{24}+7x_{25}\\
+&6x_{31}+4x_{32}+2x_{33}+7x_{34}+5x_{35}\\
+&8x_{41}+4x_{42}+2x_{43}+3x_{44}+5x_{45}\\
+&9x_{51}+10x_{52}+6x_{53}+9x_{54}+10x_{55}
\end{align}\\
\text{s.t.}\begin{cases}
\sum\limits_{j=1}^5x_{ij}=1&\forall i\in\{1,2,3,4,5\}\\
\sum\limits_{i=1}^5x_{ij}=1&\forall j\in\{1,2,3,4,5\}\quad\quad\quad\\
x_{ij}\in\{0,1\}&\forall i,j\in\{1,2,3,4,5\}
\end{cases}
$$
编写代码如下：

```python
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
```

结果如下：

```
21

     con: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
     fun: 21.0
 message: 'Optimization terminated successfully.'
     nit: 35
   slack: array([], dtype=float64)
  status: 0
 success: True
       x: array([0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
       0., 1., 0., 1., 0., 0., 0., 0.])

(array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       0, 0, 0]), 21)
```





### 4.2 例二

求解：
$$
\begin{align}
&\min z=-3x_1-2x_2-x_1\\
&\text{s.t.}\begin{cases}x_1+x_2+x_3\leqslant 7\\
4x_1+2x_2+x_3=12\\
x_1,x_2\geqslant 0\\
x_3\in\{0,1\}\end{cases}
\end{align}
$$
这是一个混合整数规划问题，可以用分支定界法求解：

```python
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
```

结果如下：

```
     con: array([0.])
     fun: -12.0
 message: 'Optimization terminated successfully.'
     nit: 2
   slack: array([1.])
  status: 0
 success: True
       x: array([0., 6., 0.])
```

