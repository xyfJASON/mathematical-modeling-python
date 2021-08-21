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



## 2 算法与代码

**非线性整数规划**没有通用的算法，事实上**非线性规划**都没有通用解法。

下列方法中，蒙特卡洛法可用于求解非线性整数规划，其余算法仅针对**线性整数规划**。

本文默认最小化目标函数，不等式约束条件均为小于等于。



### 2.1 分支定界法——整数规划（纯/混合）

忽略整数条件，单纯形法得到解作为原问题的上界。

如果当前解恰是整数解，则问题解决；否则，选取一个不满足整数条件的变量 $x_i$，设当前解中它的值为小数 $x_i^*$，则分别添加约束条件 $x\leqslant \lfloor x_i^*\rfloor$ 和 $x\geqslant \lceil x_i^*\rceil$ 得到两个新问题，分别单纯形法求解（分支）。

对于一个新问题而言，如果它是整数解，则更新原问题的下界；如果它不是整数解，若它比当前下界更优，则更新原问题的上界（定界）；否则裁剪该分支（剪枝）。然后选取一个未被裁剪的分支，重复上述步骤，直至上下界相等。



代码模板：

```python
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
```



### 2.2 隐枚举法——0-1整数规划

对暴力枚举的优化，在得到一个可行解之后，容易知道更优的解必须小于该可行解，这提供了一个强力的剪枝。



### 2.3 匈牙利算法——指派问题（0-1整数规划特例）

指派问题，可以看作 0-1 规划问题求解，也可以看作二分图匹配问题，使用匈牙利算法求解。

不过 `scipy` 已经提供了指派问题的求解函数：`scipy.optimize.linear_sum_assignment`，所以我们不必实现。

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html



### 2.4 蒙特卡洛法——非线性整数规划

蒙特卡洛法就是随机取样法，随机从可行域中取点、代入、更新答案，取点充分多后能够接近最优解。

