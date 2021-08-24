# 数据包络分析
## $C^2R$ 模型
定义多个决策单元为DMU(Decision Making Units)。有 $n$ 个 DMU，每个 DMU 有 $m$ 种投入和 $s$ 种产出，设 $x_{ij}$ 表示第 $j$ 个 DMU 的第 $i$ 种投入，$y_{ij}$ 表示第 $j$ 个 DMU 的第 $r$ 种产出，$v_i$ 表示第 $i$ 中投入的权值，$u_r$ 表示第 $r$ 种产出的权值。

向量 $X_j,Y_j$ 表示决策单元 $j$ 的输入和输出向量，$v$ 和 $u$ 表示输入、输出权值向量，有
* $X_j=(x_{1j},x_{2j},\cdots,x_{mj})$
* $Y_j=(y_{1j},y_{2j},\cdots,y_{sj})$
* $u=(u_1,u_2,\cdots,u_m)$
* $v=(v_1,v_2,\cdots,v_s)$

定义决策单元 $j$ 的效率评价质数为
$$h_j=(u^TY_j)/(v^TX_j),j=1,2,\cdots,n$$

评价决策单元 $j_0$ 效率的数学模型为
$$
\begin{aligned}
&\max\frac{u^TY_{j0}}{v^TX_{j0}}\\
&s.t.
\begin{cases}
\frac{u^TY_{j}}{v^TX_{j}}&,j=1,2,\cdots,n,\\
u\geq 0,v\geq 0&,u\neq 0,v\neq 0
\end{cases}
\end{aligned}
\tag{1}
$$
通过 Charnes-Cooper 变换：$\omega=tv,\mu=tu,t=\frac{1}{v^TX_{j0}}$，可以将上述模型转化为等价线性规划问题
$$
\begin{aligned}
&\max V_{j0}=\mu^TY_{j0},\\
&s.ts\begin{cases}
\omega^TX_j-\mu^TY_j\geq 0,j=1,2,\cdots,n,\\
\omega^TX_{j0}=1,\\
\omega\geq 0,\mu\geq 0.
\end{cases}
\end{aligned}
\tag{2}
$$
其对应的对偶问题为
$$
\begin{aligned}
&\min\theta,\\
&s.t.\begin{cases}
\mathop{\Sigma}\limits_{j=1}^n\lambda_jX_j\leq\theta X_{j0},\\
\mathop{\Sigma}\limits_{j=1}^n\lambda_jY_j\leq Y_{j0},\\
\lambda_j\geq 0,j=1,2,\cdots,n.
\end{cases}
\end{aligned}
\tag{3}
$$

* 定义 若线性规划问题的最优目标值 $V_{j0}=1$​ 则称决策单元 $j_0$​ 是弱DEA有效的。

* 定义 若线性规划问题的最优目标值 $V_{j0}=1$​，且存在最优解 $\omega>0,\mu>0$​，则称为 DEA 有效的。

## 问题 1![image-20210823155231999](C:\Users\JiangChenyang\AppData\Roaming\Typora\typora-user-images\image-20210823155231999.png)

试评价学校 A、B、C、D、E、F 哪个学校教学质量更好。

### 解析

其中每一个学校均为决策单元（DMU），有 2 个投入和 2 个产出。对应 Lingo 程序如下，其对应的线性规划问题为上述公式 $2$：


```python
model:
sets:
dmu /1..6/:s, t, p; 	!s,t,p 为中间变量;
inw/1..2/:omega;		!投入权值，无需实现确定;
outw/1..2/:mu;			!产出权值，无需实现确定;
inv(inw,dmu):x;			!投入;
outv(outw,dmu):y;		!产出;
endsets

data:
ctr=?;	!输入待评价单元;
x = 89.39 86.25 108.13 106.38 62.40 47.19
    64.3 99 99.6 96 96.2 79.9;
y = 25.2 28.2 29.4 26.4 27.2 25.2
    223 287 317 291 295 222;
enddata

max = @sum(dmu:p*t);
p(ctr) = 1;
@for(dmu(i)| i#ne#ctr: p(i)=0);
@for(dmu(j):s(j)=@sum(inw(i):omega(i)*x(i,j));
     		t(j)=@sum(outw(i):mu(i)*y(i,j));
            s(j)>t(j));
@sum(dmu:p*s)=1;
end
```

程序运行结果：

| 学校 | 最优目标值 $V_{j0}$ |
| ---- | ------------------- |
| A    | 1                   |
| B    | 0.9096              |
| C    | 0.9635              |
| D    | 0.9143              |
| E    | 1                   |
| F    | 1                   |

发现学校A、E、F最优目标值为 1，且变量 $\omega,\mu$​ 大于0，因此其是 DEA 有效的。
