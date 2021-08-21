<h1 style="text-align:center"> 非线性规划 </h1>
<h2 style="text-align:center"> Nonlinear Programming </h2>
<div style="text-align:center"> xyfJASON </div>



## 1 概述

若目标函数或约束条件包含非线性函数，则称这种规划问题是非线性规划问题。

没有通用的算法，各个方法都有自己特定的使用范围。



---



## 2 算法与代码

使用 `scipy.optimize.minimize`，提供了众多优化方法。

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html



|     方法     | 约束条件 |                 使用算法                  | 是否需要<br>梯度向量<br>海塞矩阵 |                             备注                             |
| :----------: | :------: | :---------------------------------------: | :------------------------------: | :----------------------------------------------------------: |
|      CG      |  无约束  |      nonlinear conjugate<br>gradient      |              是；否              |                                                              |
|     BFGS     |  无约束  |               quasi-Newton                |              是；否              |   在非光滑优化上都能有很好的表现<br>同时返回近似海塞逆矩阵   |
|  Newton-CG   |  无约束  |             truncated Newton              |              是；是              |                        适合大规模问题                        |
|    dogleg    |  无约束  |           dog-leg trust-region            |      是；是<br>（要求正定）      |                                                              |
|  trust-ncg   |  无约束  | Newton conjugate<br>gradient trust-region |              是；是              |                        适合大规模问题                        |
| trust-krylov |  无约束  |        Newton GLTR<br>trust-region        |              是；是              |           适合大规模问题<br>中规模和大规模推荐使用           |
| trust-exact  |  无约束  |                trust-exact                |              是；是              |                  小规模和中规模最为推荐使用                  |
| Nelder-Mead  | 边界约束 |                  Simplex                  |              否；否              |          适用于许多应用<br>不如要求导函数的方法精确          |
|   L-BFGS-B   | 边界约束 |                 L-BFGS-B                  |              是；否              |                    同时返回近似海塞逆矩阵                    |
|    Powell    | 边界约束 |            conjugate direction            |              否；否              |                     目标函数无需可导<br>                     |
|     TNC      | 边界约束 |             truncated Newton              |              是；否              |        打包了 C 语言实现<br>Newton-CG 的边界约束版本         |
|    COBYLA    |  有约束  |                  COBYLA                   |              否；否              |   打包了 FORTRAN 语言实现<br>只支持不等式（大于等于）约束    |
|    SLSQP     |  有约束  |  Sequential Least<br>SQuares Programming  |  是；否<br>还需要约束条件的梯度  |                                                              |
| trust-constr |  有约束  |               trust-region                |              是；是              | 根据问题自动在两种方法中切换<br>最多功能的约束优化实现<br>用于大规模问题的最适合方法 |



---



## 3 例题



### 3.1 例一

求函数 $f(x)=100(x_2-x_1^2)^2+(1-x_1)^2$ 的极小值。

这是无约束问题，梯度和海塞矩阵都比较好算，不妨使用 `trust-exact` 方法：
$$
\grad f(x)=\begin{bmatrix}-400(x_2-x_1^2)x_1-2(1-x_1)\\200(x_2-x_1^2)\end{bmatrix}
$$

$$
Hessian(f)=\begin{bmatrix}
-400(x_2-x_1^2)+800x_1^2+2&-400x_1\\
-400x_1&200
\end{bmatrix}
$$

```python
def f(x):
    return 100*(x[1]-x[0]*x[0])**2+(1-x[0])**2

def grad(x):
    g = np.zeros(2)
    g[0] = -400*(x[1]-x[0]*x[0])*x[0]-2*(1-x[0])
    g[1] = 200*(x[1]-x[0]*x[0])
    return g

def hessian(x):
    h = np.zeros((2, 2))
    h[0, 0] = -400*(x[1]-x[0]*x[0])+800*x[0]*x[0]+2
    h[0, 1] = -400 * x[0]
    h[1, 0] = -400 * x[0]
    h[1, 1] = 200
    return h

res = minimize(fun=f,
               x0=np.zeros(2),
               method='trust-exact',
               jac=grad,
               hess=hessian)
print(res)
```

结果如下：

```
     fun: 1.1524542015768823e-13
    hess: array([[ 801.99967846, -399.99991693],
       [-399.99991693,  200.        ]])
     jac: array([ 1.03264509e-05, -5.37090168e-06])
 message: 'Optimization terminated successfully.'
    nfev: 18
    nhev: 18
     nit: 17
    njev: 15
  status: 0
 success: True
       x: array([0.99999979, 0.99999956])
```



### 3.2 例二

求 $f(x)=(x-3)^2-1, x\in[0,5]$ 的最小值。

计算梯度：
$$
\grad f(x)=2(x-3)
$$
这是有边界约束问题，可以使用 L-BFGS-B 方法：

```python
def f(x):
    return (x-3)**2-1

def grad(x):
    return 2*(x-3)

res = minimize(fun=f,
               x0=np.array([0]),
               method='L-BFGS-B',
               jac=grad,
               bounds=[(0, 5)])
print(res)
```

结果如下：

```
      fun: array([-1.])
 hess_inv: <1x1 LbfgsInvHessProduct with dtype=float64>
      jac: array([0.])
  message: 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
     nfev: 3
      nit: 2
     njev: 3
   status: 0
  success: True
        x: array([3.])
```



### 3.3 例三

已知 $f(x)=e^{x_1}(4x_1^2+2x_2^2+4x_1x_2+2x_2+1)$，求
$$
\begin{align}
&\min f(x)\\
&\text{s.t.}\begin{cases}
x_1x_2-x_1-x_2\leqslant -1.5\\
x_1x_2\geqslant -10
\end{cases}
\end{align}
$$
能算梯度，先把梯度算出来：
$$
\grad f(x)=
\begin{bmatrix}
e^{x_1}(4x_1^2+2x_2^2+4x_1x_2+8x_1+6x_2+1)\\
e^{x_1}(4x_1+4x_2+2)
\end{bmatrix}
$$
这是有约束问题，可以使用 SLSQP 方法：

```python
def f(x):
    return np.e ** x[0] * (4 * x[0] * x[0] + 2 * x[1] * x[1] + 4 * x[0] * x[1] + 2 * x[1] + 1)

def grad(x):
    g = np.zeros(2)
    g[0] = np.e ** x[0] * (4 * x[0] * x[0] + 2 * x[1] * x[1] + 4 * x[0] * x[1] + 8 * x[0] + 6 * x[1] + 1)
    g[1] = np.e ** x[0] * (4 * x[0] + 4 * x[1] + 2)
    return g

def get_constr():
    def constr_f1(x):
        return x[0] + x[1] - x[0] * x[1] - 1.5

    def constr_grad1(x):
        return np.array([1 - x[1], 1 - x[0]])

    def constr_f2(x):
        return x[0] * x[1] + 10

    def constr_grad2(x):
        return np.array([x[1], x[0]])

    c = [
        dict(type='ineq',
             fun=constr_f1,
             jac=constr_grad1),
        dict(type='ineq',
             fun=constr_f2,
             jac=constr_grad2)
        ]
    return c

constr = get_constr()
res = minimize(fun=f,
               x0=np.array([-2, 2]),
               method='SLSQP',
               jac=grad,
               constraints=constr)
print(res)
```

结果如下：

```
     fun: 0.02355037962417156
     jac: array([ 0.01839705, -0.00228436])
 message: 'Optimization terminated successfully'
    nfev: 9
     nit: 8
    njev: 8
  status: 0
 success: True
       x: array([-9.54740503,  1.04740503])
```

