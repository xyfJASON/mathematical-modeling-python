<h1 style="text-align:center"> 微分方程 </h1>
<h2 style="text-align:center"> Differential Equation </h2>
<div style="text-align:center"> xyfJASON </div>



## 1 实例·三级火箭



### 1.1 一级火箭不可行

1. 计算卫星进入 600km 高空轨道，火箭需要的最低速度

   前提假设：

   1. 卫星以地球引力为向心力做匀速圆周运动
   2. 地球是质量在球心的均匀球体
   3. 忽略其余天体的引力影响

   根据高中物理知识，有黄金代换：
   $$
   GM=R^2g
   $$
   又引力提供向心力：
   $$
   \frac{GMm}{r^2}=\frac{mv^2}{r}
   $$
   解得：
   $$
   v=R\sqrt{\frac{g}{r}}
   $$
   其中，$g$ 为重力加速度，$R$ 为地球半径，$r$ 为轨道半径，求得：$v\approx 7.6\text{km/s}$。

2. 计算火箭推进力和升空速度

   前提假设：

   1. 火箭做直线运动，忽略自身重力和空气阻力
   2. 火箭的质量、速度为时间 $t$ 的函数 $m(t),v(t)$
   3. 火箭喷出气体的速度相对火箭本身为常数 $u$

   考虑 $(t,t+\Delta t)$ 微元时间内，火箭质量变化量为：
   $$
   m(t+\Delta t)-m(t)=\frac{\mathrm dm}{\mathrm dt}\Delta t+o(\Delta t)
   $$
   喷出气体相对于地面的速度为 $v(t)-u$​，故由动量守恒有：
   $$
   m(t+\Delta t)v(t+\Delta t)-m(t)v(t)=\left[\frac{\mathrm dm}{\mathrm dt}\Delta t+o(\Delta t)\right](v(t)-u)
   $$
   联立得：
   $$
   m\frac{\mathrm dv}{\mathrm dt}=-u\frac{\mathrm dm}{\mathrm dt}=F_\text{推}
   $$
   解得：
   $$
   v(t)=-u\ln m(t)+C
   $$
   代入初值条件：$m(0)=m_0,\,v(0)=v_0$，解得：
   $$
   v(t)=v_0+u\ln\frac{m_0}{m(t)}
   $$

3. 一级火箭速度上限

   前提假设：

   1. 设火箭质量分为三部分：有效质量 $m_p$​，燃料质量 $m_F$​，结构质量 $m_s$​
   2. 初速度 $v_0=0$
   3. 根据目前技术条件，设 $u=3\text{km/s}$​，$\dfrac{m_F}{m_s}\leqslant 8$​​​​​

   根据上一部分的推导，当燃料耗尽时（$m_F=0$​），末速度为：
   $$
   v=u\ln \frac{m_p+m_s+m_F}{m_p+m_s}\leqslant 3\cdot \ln 9\approx6.6\text{km/s}
   $$
   因此无法达到所需速度。



### 1.2 理想火箭

理想火箭指能够随时抛去多余结构质量的火箭，假设在 $(t,t+\Delta t)$ 时间内火箭质量减少量中抛弃结构质量占比 $\alpha$，燃料燃烧质量占比 $1-\alpha$，则有新的动量守恒式：
$$
m(t+\Delta t)v(t+\Delta t)-m(t)v(t)=\alpha\frac{\mathrm dm}{\mathrm dt}\Delta t\cdot v(t)+(1-\alpha)\frac{\mathrm dm}{\mathrm dt}\Delta t\cdot (v(t)-u)+o(\Delta t)
$$
联立得：
$$
m\frac{\mathrm dv}{\mathrm dt}=(\alpha-1)u\frac{\mathrm dm}{\mathrm dt}
$$
解得：
$$
v(t)=v_0+(1-\alpha)u\ln \frac{m_0}{m(t)}
$$
当燃料耗尽并且结构质量抛弃完时（$m_s=m_F=0$​），同样假设 $v_0=0$，有：
$$
v=(1-\alpha)u\ln\frac{m_0}{m_p}
$$
因此只要 $m_0$ 足够大，就能使火箭达到需要的速度。



### 1.3 三级火箭

理想火箭虽然无法实现，但为我们指明了方向。多级火箭指从末级开始逐级燃烧，第 $i$​ 级火箭燃烧殆尽后自动脱离，同时第 $i+1$​ 级火箭自动点火。

前提假设：第 $i$ 级火箭结构质量为 $\lambda m_i$，燃料质量为 $(1-\lambda)m_i$，$\lambda$​ 对各级火箭均相等

以二级火箭为例，当第一级火箭燃烧殆尽时，其速度为：
$$
v_1=u\ln\frac{m_1+m_2+m_p}{\lambda m_1+m_2+m_p}
$$
第二级火箭燃烧殆尽时，其速度为：
$$
v_2=v_1+u\ln\frac{m_2+m_p}{\lambda m_2+m_p}=u\ln\left[\frac{m_1+m_2+m_p}{\lambda m_1+m_2+m_p}\cdot\frac{m_2+m_p}{\lambda m_2+m_p}\right]
$$
同理，一个 $n$​ 级火箭在第 $k$​ 级燃烧殆尽时，其速度为：
$$
v_k=u\ln\prod_{i=1}^k\frac{m_i+m_{i+1}+\cdots+m_n+m_p}{\lambda m_i+m_{i+1}+\cdots+m_n+m_p}
$$



---



## 2 实例·人口模型



### 2.1 Malthus 模型

模型假设：

1. 人口数 $x(t)$ 是关于时间 $t$ 的连续可导函数
2. 人口增长率 $r$ 是常数
3. 没有人口流动

由假设，$(t,t+\Delta t)$ 时间内人口增量为：
$$
x(t+\Delta t)-x(t)=\frac{\mathrm dx(t)}{\mathrm dt}\Delta t+o(\Delta t)=x(t)\cdot r\cdot \Delta t
$$
于是有：
$$
\frac{\mathrm dx(t)}{\mathrm dt}=r\cdot x(t)
$$
代入初值条件 $x(0)=x_0$，解得：
$$
x(t)=x_0e^{rt}
$$
这个模型与 1700-1961 年的世界人口较为符合，但是显然不会一直适用下去。



### 2.2 Logistic 模型

修正上述人口增长率 $r$ 为人口数 $x$ 的减函数：$r=r(x)$，即人口越多，增长率越慢，这是符合实际的。

假设 $r$ 在人口数达到上限 $x_m$ 之前是关于 $x$ 的线性函数，在达到上限后是 $0$，那么有：
$$
r(x)=\begin{cases}
r_0\left(1-\dfrac{x}{x_m}\right)&0\leqslant x\leqslant x_m\\
0&x>x_m
\end{cases}
$$
代入微分方程得到：
$$
\frac{\mathrm dx(t)}{\mathrm dt}=r(x)\cdot x(t)=r_0\left(1-\dfrac{x(t)}{x_m}\right)x(t)
$$
分离变量法解上述微分方程，并代入初值条件 $x(0)=x_0$，得：
$$
x(t)=\frac{x_m}{1+\left(\dfrac{x_m}{x_0}-1\right)e^{-rt}}
$$

> 函数 $y=\dfrac{1}{1+e^{-x}}$ 称为 Logistic 函数，故称该模型为 Logistic 模型，是一条 S 形曲线。



---



## 3 实例·放射性废料处理

以往放射性废料的处理方法是装入圆桶并沉入海底。已知圆桶质量 $m=239.46\text{kg}$，体积 $V=0.2058\text{m}^3$，海水密度 $\rho=1035.71\text{kg}/\text{m}^3$​，海底深度 $90\text{m}$。若圆桶沉底时速度超过 $12.2\text{m/s}$ 就会破裂。



### 3.1 $f=kv$

假若水的阻力与速度成正比且正比例系数 $k=0.6$​。

设圆桶所受合力为 $F$，阻力为 $f$，那么我们知道：
$$
\begin{cases}
F=mg-\rho gV-f&\text{合力}=\text{重力}-\text{浮力}-\text{阻力}\\
f=kv&\text{阻力}\sim\text{速度}\\
F=ma=m\dfrac{\mathrm dv}{\mathrm dt}&\text{牛顿第二定律}
\end{cases}
$$
整理得：
$$
m\frac{\mathrm dv}{\mathrm dt}=mg-\rho gV-kv
$$
这是一个一阶线性微分方程，其解为：
$$
\begin{align}
v(t)&=e^{-\int\frac{k}{m}\mathrm dt}\left(\int \left(g-\frac{\rho gV}{m}\right)e^{\int\frac{k}{m}\mathrm dt}\mathrm dt+C\right)\\
&=Ce^{-\frac{kt}{m}}+\frac{g(m-\rho V)}{k}
\end{align}
$$
代入初值条件 $v(0)=0$，代入已知数值，得到：
$$
v(t)=429.744-429.744e^{-0.00250564t}
$$
又因为 $s=\int v\mathrm dt$，所以：
$$
s(t)=-171511.0+429.744t+171511.0e^{-0.00250564t}
$$
当 $s=90$​ 时，求根得 $t=12.9994\text{s}$​，于是计算得：$v(12.9994)=13.7720\text{m/s}>12.2\text{m/s}$​​。不可行。



### 3.2 $f=kv^2$

同理可得微分方程：
$$
m\frac{\mathrm dv}{\mathrm dt}=mg-\rho gV-kv^2
$$
加上初始条件解得：
$$
v(t)=20.7303\tanh(0.0519t)
$$
若要使得 $v(t)<12.2$，则 $T<13.0025$，计算得到 $s(T)=\int_0^Tv(t)\mathrm dt=84.8439\text{m}<90\text{m}$​。不可行。



---



## 4 算法与代码

`scipy.integrate.solve_ivp` 提供了求常微分方程初值问题数值解的若干算法。

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html



|  方法  |                           使用算法                           |                   备注                    |
| :----: | :----------------------------------------------------------: | :---------------------------------------: |
|  RK45  |          Explicit Runge-Kutta method of order 5(4)           |               适用于非刚性                |
|  RK23  |          Explicit Runge-Kutta method of order 3(2)           |               适用于非刚性                |
| DOP853 |            Explicit Runge-Kutta method of order 8            |     适用于非刚性<br>精度要求高时推荐      |
| Radau  | Implicit Runge-Kutta method of the Radau IIA family of order 5 |                适用于刚性                 |
|  BDF   |      Implicit multi-step variable-order (1 to 5) method      |                适用于刚性                 |
| LSODA  |                       Adams/BDF method                       | 自适应刚性/非刚性<br>用起来可能不是很方便 |

>  Trick：先用 RK45 跑，如果迭代过多、发散或失败，那么问题很有可能是刚性的，切换 Radau 或者 BDF。



---



## 5 例题



### 5.1 例一

求解：
$$
y'=-2y+2x^2+2x,\,0\leqslant x\leqslant0.5,\,y(0)=1
$$
这是一个初值问题，可用 RK45 方法：

```python
def f(x, y):
    return -2 * y + 2 * x * x + 2 * x

x_range = (0, 0.5)
y0 = np.array([1])

res = solve_ivp(fun=f,
                t_span=x_range,
                y0=y0,
                method='RK45',
                t_eval=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
print(res)
```

结果如下：

```
  message: 'The solver successfully reached the end of the integration interval.'
     nfev: 14
     njev: 0
      nlu: 0
      sol: None
   status: 0
  success: True
        t: array([0. , 0.1, 0.2, 0.3, 0.4, 0.5])
 t_events: None
        y: array([[1.        , 0.82869232, 0.70996243, 0.63841545, 0.60929247,
        0.61811694]])
 y_events: None
```



### 5.2 例二

求解：
$$
y'''-3y''-y'y=0,\,y(0)=0,\,y'(0)=1,\,y''(0)=-1
$$
**高阶微分方程必须做变量代换，化成一阶微分方程组**才能用 `scipy` 求解。

设 $y_1=y,\,y_2=y',\,y_3=y''$，则上述问题可以转换为：
$$
\begin{cases}
y_1'=y_2&y_1(0)=0\\
y_2'=y_3&y_2(0)=1\\
y_3'=3y_3+y_1y_2&y_3(0)=-1
\end{cases}
$$
这是一个初值问题，可以用 RK45 方法：

```python
def f(_, y):
    return np.array([y[1], y[2], 3*y[2]+y[0]*y[1]])

x_range = (0, 1)
y0 = np.array([0, 1, -1])

res = solve_ivp(fun=f,
                t_span=x_range,
                y0=y0,
                method='RK45',
                t_eval=[0, 0.5, 1])
print(res)
```

结果如下：

```
  message: 'The solver successfully reached the end of the integration interval.'
     nfev: 44
     njev: 0
      nlu: 0
      sol: None
   status: 0
  success: True
        t: array([0. , 0.5, 1. ])
 t_events: None
        y: array([[  0.        ,   0.28216608,  -0.75866533],
       [  1.        ,  -0.14284572,  -5.24327063],
       [ -1.        ,  -4.38890227, -19.44348305]])
 y_events: None
```

