<h1 style="text-align:center"> 拟合 </h1>
<h2 style="text-align:center"> Fitting </h2>
<div style="text-align:center"> xyfJASON </div>



## 1 线性最小二乘法

给定平面上 $n$ 个点 $(x_i, y_i)$，求函数 $y=f(x)$，使得 $f(x)$ 与这些数据点最为接近。

基本思路：取一组 $[a,b]$​​ 上线性无关的函数 $\varphi_1(x),\varphi_2(x),\dots,\varphi_m(x)$​​，设 $\Phi=\text{Span}\{\varphi_1(x),\varphi_2(x),\ldots,\varphi_m(x)\}$​​，求解 $\varphi^*(x)\in\Phi$​​​ 使得：
$$
\varphi^*(x)=\mathop{\arg\min}_{\varphi\in\Phi}\sum_{i=1}^n\left[y_i-\varphi(x_i)\right]^2
$$
设 $R=\begin{bmatrix}\varphi_1(x_1)&\cdots&\varphi_m(x_1)\\\vdots&\ddots&\vdots\\\varphi_1(x_n)&\cdots&\varphi_m(x_n)\\\end{bmatrix}\in\R^{n\times m}$，$A=\begin{bmatrix}a_1\\\vdots\\a_m\end{bmatrix}\in\R^{m}$，$Y=\begin{bmatrix}y_1\\\vdots\\y_n\end{bmatrix}\in\R^n$，求解该问题相当于求解：
$$
\min\quad J(a_1,a_2,\ldots,a_m)=\sum_{i=1}^n\left[y_i-\sum_{j=1}^ma_j\varphi_j(x_i)\right]^2=||RA-Y||_2^2
$$
求偏导并令其为零：
$$
\frac{\part J}{\part a_k}=2\sum_{i=1}^n\varphi_k(x_i)\left[\sum_{j=1}^ma_j\varphi_j(x_i)-y_i\right]=0,\quad k=1,2,\ldots,m
$$
解得：
$$
\sum_{j=1}^ma_j\left[\sum_{i=1}^n\varphi_k(x_i)\varphi_j(x_i)\right]=\sum_{i=1}^ny_i\varphi_k(x_i),\quad k=1,2,\ldots,m
$$
即：
$$
R^TRA=R^TY
$$
当 $\{\varphi_1(x),\varphi_2(x),\ldots,\varphi_m(x)\}$ 线性无关时，$R$ 列满秩，$R^TR$ 可逆，于是方程有唯一解：
$$
A=\left(R^TR\right)^{-1}R^TY
$$


---



## 2 线性最佳平方逼近

最佳平方逼近其实就是连续情况下的最小二乘法，推理与最小二乘法类似，仅需将求和改为积分。

如果定义内积：
$$
\begin{align}
(\varphi_j,\varphi_k)&=\sum_{i=1}^n\varphi_j(x_i)\varphi_k(x_i)&&\text{discrete}\\
&=\int_a^b\varphi_j(x)\varphi_i(x)\mathrm dx&&\text{continuous}
\end{align}
$$
那么上一节的方程 $R^TRA=R^TY$ 可以写作：
$$
\begin{bmatrix}(\varphi_1,\varphi_1)&\cdots&(\varphi_1,\varphi_m)\\\vdots&\ddots&\vdots\\(\varphi_m,\varphi_1)&\cdots&(\varphi_m,\varphi_m)\end{bmatrix}\begin{bmatrix}a_1\\\vdots\\a_m\end{bmatrix}=\begin{bmatrix}(y,\varphi_1)\\\vdots\\(y,\varphi_m)\end{bmatrix}
$$
当 $\varphi_1,\varphi_2,\ldots,\varphi_m$ 线性无关时，上述矩阵非奇异，方程有唯一解。



---



## 3 代码

可以看到，所谓曲线拟合，最后其实都归约到了优化问题，只不过对于线性最小二乘/线性最佳平方逼近而言，我们可以直接得到这个优化问题的解析解。所以 `scipy` 中这部分的接口放在了 `scipy.optimize` 下。



### 3.1 线性最小二乘优化

使用 `scipy.optimize.lsq_linear`，其规定的标准形式为：
$$
\begin{align}
&\min\ \frac12||Ax-b||^2\\
&\text{s.t.}\ \mathrm{lb}\leqslant x\leqslant \mathrm{ub}
\end{align}
$$
Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html

<br>

使用 `scipy.optimize.nnls` 可以解决无约束条件的非负解，其标准形式为：
$$
\mathop{\arg\min}_x\,||Ax-b||_2,\quad x\geqslant 0
$$
Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html

<br>



### 3.2 非线性最小二乘优化

使用 `scipy.optimize.least_squares`，其规定的标准形式为：
$$
\begin{align}
&\min\ F(x)=\frac12\times\sum_{i=0}^{n-1}J(f_i^2(x))\\
&\text{s.t.}\ \mathrm{lb}\leqslant x\leqslant\mathrm{rb}
\end{align}
$$
其中 $f_i$​​ 读入一个 $m$​​ 维向量 $x$​​（即我们要优化的参数），输出一个 $n$​​ 维向量 $f_i(x)$​​，例如在线性最小二乘中，$f_i$​​ 就是 $||A_{i,\bullet}x-b||$​​；

$J$​ 是损失函数（例如线性、soft-l1 等等），线性损失 $J(x)=x$ 即原始的最小二乘；

$F$ 是代价函数，我们的目标是最小化代价函数。

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html



### 3.3 曲线拟合

`scipy.optimize.curve_fit` 使用非线性最小二乘优化拟合曲线。

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

输入用来拟合的（非）线性函数 `f(x, *params)`、数据点 $(x_i,y_i)$​，输出拟合的参数 `params`。参数可有约束。



---



## 4 例题



### 4.1 例一

某乡镇企业 1990 年 - 1996 年的生产利润如下表所示：

| 年份      | 1990 | 1991 | 1992 | 1993 | 1994 | 1995 | 1996 |
| --------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 利润/万元 | 70   | 122  | 144  | 152  | 174  | 196  | 202  |

试预测 1997 年和 1998 年的利润。

作散点图观察，发现年生产利润几乎直线上升，因此可用 $y=a_0+a_1x$ 作拟合，代码如下：

```python
x = np.array([1990, 1991, 1992, 1993, 1994, 1995, 1996])
y = np.array([70, 122, 144, 152, 174, 196, 202])
A = np.stack((np.ones(x.shape[0]), x), axis=1)
b = y

res = np.linalg.inv(A.T @ A) @ A.T @ y
print('#1:', res)

res = lsq_linear(A, b)
print('#2:', res)

res = curve_fit(lambda x, a0, a1: a0 + a1 * x, x, y)
print('#3:', res)
```

用了 3 种计算方式，第一种是直接用第一节的结论计算，第二种是调用 `lsq_linear` 函数，第三种是调用 `curve_fit` 函数，结果如下：

```
#1: [-4.07050714e+04  2.05000000e+01]
#2:  active_mask: array([0., 0.])
        cost: 419.3571428571963
         fun: array([ 19.92857143, -11.57142857, -13.07142857,  -0.57142857,
        -2.07142857,  -3.57142857,  10.92857143])
     message: 'The unconstrained solution is optimal.'
         nit: 0
  optimality: 4.353933036327362e-08
      status: 3
     success: True
           x: array([-4.07050714e+04,  2.05000000e+01])
#3: (array([-4.07050714e+04,  2.05000000e+01]), array([[ 2.37958401e+07, -1.19396970e+04],
       [-1.19396970e+04,  5.99081633e+00]]))
```

可以看见 3 种方式计算结果是一致的，但第三种方式最为方便且应用面最为广泛。



### 4.2 例二

用 $y=ae^{bx}$​​ 拟合数据：

| x    | 1    | 2    | 3    | 4    | 5    | 6    | 7     | 8     |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ----- |
| y    | 15.3 | 20.5 | 27.4 | 36.6 | 49.1 | 65.6 | 87.87 | 117.6 |

直接拟合可以看作非线性问题，也可以取对数以后变为线性问题，代码如下：

```python
x = np.arange(1, 9)
y = np.array([15.3, 20.5, 27.4, 36.6, 49.1, 65.6, 87.87, 117.6])

res = curve_fit(lambda x, a, b: a * np.exp(b * x), x, y)
print(res)

res = curve_fit(lambda x, a, b: np.log(a) + b * x, x, np.log(y))
print(res)
```

结果如下：

```
(array([11.4250665 ,  0.29142395]), array([[ 1.40603526e-04, -1.72314524e-06],
       [-1.72314524e-06,  2.21398094e-08]]))
(array([11.4357665 ,  0.29126345]), array([[ 7.77920161e-05, -1.20044445e-06],
       [-1.20044445e-06,  2.33272888e-08]]))
```

从协方差矩阵可见，转化为线性问题之后误差稍小一些。

