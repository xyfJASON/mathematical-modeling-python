<h1 style="text-align:center"> 线性规划 </h1>
<h2 style="text-align:center"> Linear Programming </h2>
<div style="text-align:center"> xyfJASON </div>



## 1 标准型

在一组**线性约束条件**下，最大（小）化**线性目标函数**：
$$
\begin{align}
&\min_x\ c^Tx\\
&\text{s.t.}\begin{cases}
A_{ub}x\leqslant b_{ub}\\
A_{eq}x=b_{eq}\\
l\leqslant x\leqslant u
\end{cases}
\end{align}
$$

> 注：线性规划问题可以求最大、求最小，约束条件可以是小于等于、大于等于，为统一标准，`scipy` 规定标准形式如上，在使用 `scipy` 求解时，首先应化成上述形式。



---



## 2 变式



### 2.1 含有绝对值

例如：
$$
\begin{align}
&\min_x\ |x_1|+|x_2|+\cdots+|x_n|\\
&\text{s.t.}\ Ax\leqslant b
\end{align}
$$
只需注意到：对任意的 $x_i$​，存在 $u_i,v_i\geqslant 0$​ 满足：
$$
\begin{cases}
|x_i|=u_i+v_i\\
x_i=u_i-v_i
\end{cases}
$$
（事实上，取 $u_i=(|x_i|+x_i)/2,\,v_i=(|x_i|-x_i)/2$​ 即可）

于是原问题转化为：
$$
\begin{align}
&\min_{u,v} \sum_{i=1}^n(u_i+v_i)\\
&\text{s.t.}\begin{cases}A(u-v)\leqslant b\\
u,v\geqslant 0
\end{cases}
\end{align}
$$
进一步改写为：
$$
\begin{align}
&\min \sum_{i=1}^n(u_i+v_i)\\
&\text{s.t.}\begin{cases}[A,-A]\begin{bmatrix}u\\v\end{bmatrix}\leqslant b\\
\begin{bmatrix}u\\v\end{bmatrix}\geqslant 0
\end{cases}
\end{align}
$$




### 2.2 $\min\max$​

例如：
$$
\min_x\max_yf(x,y)
$$
只需要引入新的变量：$u=\max\limits_yf(x,y)$​，​那么一定有：$f(x,y)\leqslant u$​，于是问题转化为：
$$
\begin{align}
&\min_x u\\
&\text{s.t.}\ f(x,y)\leqslant u
\end{align}
$$



---



## 3 算法与代码

求解线性规划的算法有很多，例如单纯形法、改进单纯形法、内点法等等。

我们使用 `scipy.optimize.linprog` 即可。

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html



---



## 4 例题



### 4.1 例一

$$
\begin{align}
&\min z=2x_1+3x_2+x_3\\
&\text{s.t.}\begin{cases}x_1+4x_2+2x_3\geqslant 8\\
3x_1+2x_2\geqslant 6\\
x_1,x_2,x_3\geqslant 0
\end{cases}
\end{align}
$$

编写代码如下：

```python
c = np.array([2, 3, 1])
A_ub = np.array([[-1, -4, -2],
                 [-3, -2, 0]])
b_ub = np.array([-8, -6])
res = linprog(c=c,
              A_ub=A_ub,
              b_ub=b_ub,
              bounds=[(0, None), (0, None), (0, None)],
              method='revised simplex')
print(res)
```

结果如下：

```
     con: array([], dtype=float64)
     fun: 7.0
 message: 'Optimization terminated successfully.'
     nit: 1
   slack: array([0., 0.])
  status: 0
 success: True
       x: array([2., 0., 3.])
```



### 4.2 例二

$$
\begin{align}
&\min z=|x_1|+2|x_2|+3|x_3|+4|x_4|\\
&\text{s.t.}\begin{cases}x_1-x_2-x_3+x_4\leqslant -2\\
x_1-x_2+x_3-3x_4\leqslant -1\\
x_1-x_2-2x_3+3x_4\leqslant -1/2
\end{cases}
\end{align}
$$

做变量代换：$u_i=(x_i+|x_i|)/2,\,v=(|x_i|-x_i)/2$，则原问题转化为：
$$
\begin{align}
&\min z=u_1+v_1+2u_2+2v_2+3u_3+3v_3+4u_4+4v_4\\
&\text{s.t.}\begin{cases}u_1-v_1-u_2+v_2-u_3+v_3+u_4-v_4\leqslant -2\\
u_1-v_1-u_2+v_2+u_3-v_3-3u_4+3v_4\leqslant -1\\
u_1-v_1-u_2+v_2-2u_3+2v_3+3u_4-3v_4\leqslant -1/2\\
u_1,v_1,u_2,v_2,u_3,v_3,u_4,v_4\geqslant 0
\end{cases}
\end{align}
$$
编写代码如下：

```python
c = np.array([1, 2, 3, 4, 1, 2, 3, 4])
A_ub = np.array([[1, -1, -1, 1, -1, 1, 1, -1],
                 [1, -1, 1, -3, -1, 1, -1, 3],
                 [1, -1, -2, 3, -1, 1, 2, -3]])
b_ub = np.array([-2, -1, -1/2])
res = linprog(c=c,
              A_ub=A_ub,
              b_ub=b_ub,
              bounds=[(0, None)] * 8,
              method='revised simplex')
print(res)
```

结果如下：

```
     con: array([], dtype=float64)
     fun: 2.0
 message: 'Optimization terminated successfully.'
     nit: 6
   slack: array([0. , 1. , 1.5])
  status: 0
 success: True
       x: array([0., 0., 0., 0., 2., 0., 0., 0.])
```



### 4.3 例三

$$
\min_{x_i}\max_{y_i}|x_i-y_i|
$$

令 $z=\max\limits_{y_i}|x_i-y_i|$​，则 $|x_i-y_i|\leqslant z$​，即 $-z\leqslant x_i-y_i\leqslant z$，于是原问题转换为：
$$
\begin{align}
&\min_x z\\
&\text{s.t.}\begin{cases}
x_1-y_1\leqslant z,\ldots,x_n-y_n\leqslant z\\
y_1-x_1\leqslant z,\ldots,y_n-x_n\leqslant z
\end{cases}
\end{align}
$$
成为一个线性规划问题。

