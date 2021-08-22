# 主成分分析
## 基本思想与方法
### 基本思想
考虑随机变量 $X=(X_1,X_2,\cdots,X_p)$​ 与权重 $c_1,c_2,\cdots,c_p$​，寻找向量 $C=(c_1,c_2,\cdots,c_p)$​ 使得
$$\max s=X\cdot C=\mathop{\Sigma}\limits_{i=1}^pc_i\cdot X_i\\
|C|=1$$​
其中 $Z=C\cdot X$​ 称为主成分。寻找若干向量 $C_i$​ 使得 $C_i$​ 满足上述条件且任意 $C_i$​ 之间正交，从中选取 $k$​ 个向量，可以将原来 $p$​ 维数据降至 $k$​ 维，从而便于数据挖掘与分析

### 方法
设有多元随机变量 $X$ 有 $p$ 个指标变量 $x_1,x_2,\cdots,x_p$，其在第 $i$ 次试验中取值为
$$a_{i1},a_{i2},\cdots,a_{ip},i=1,2,\cdots,n$$
将其写成矩阵形式
$$
A=\begin{bmatrix}
a_{11}&a_{12}&\cdots&a_{1p}\\
a_{21}&a_{22}&\cdots&a_{2p}\\
\vdots&\vdots&\ddots&\vdots\\
a_{n1}&a_{n2}&\cdots&a_{np}\\
\end{bmatrix}
$$
其中 $A^TA$ 的特征向量即为主成分系数 $C$（限于时间有限，没有深入了解推导过程），保留部分特征向量，通常约定保留的特征向量对应特征值之和占总特征值之和的 85% 以上。

也可以对数据进行标准化，将标准化数据矩阵记为 $\mathop{A}\limits^\sim$，随后得到相关系数矩阵 $R={\mathop{A}\limits^\sim}^T\mathop{A}\limits^\sim/(n-1)$，接着计算相关系数矩阵 $R$ 的特征值和特征向量即可。

除上述 85% 指标外，有时也需要考虑其对原变量 $x_i$ 的贡献值，主成分 $z_j$ 对原变量 $x_i$ 的贡献值为
$$\rho=\mathop{\Sigma}\limits_{j=1}^rr^2(z_j,x_i)$$

## 案例
### Hald 水泥问题（主成分回归分析）

考察含四种成分化学 x1,x2,x3,x4 的水泥，每一克释放的热量 y 与四种含量之间的关系数据共 13 组，建立 y 与四种化学成分的函数关系。


```python
import numpy as np
import pandas as pd
df = pd.read_csv('./data/sn.txt', sep='\t', header=None)
df.columns = ['x1', 'x2', 'x3', 'x4', 'y']
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>26</td>
      <td>6</td>
      <td>60</td>
      <td>78.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>29</td>
      <td>15</td>
      <td>52</td>
      <td>74.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11</td>
      <td>56</td>
      <td>8</td>
      <td>20</td>
      <td>104.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>31</td>
      <td>8</td>
      <td>47</td>
      <td>87.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>52</td>
      <td>6</td>
      <td>33</td>
      <td>95.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11</td>
      <td>55</td>
      <td>9</td>
      <td>22</td>
      <td>109.2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>71</td>
      <td>17</td>
      <td>6</td>
      <td>102.7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>31</td>
      <td>22</td>
      <td>44</td>
      <td>72.5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>54</td>
      <td>18</td>
      <td>22</td>
      <td>93.1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>21</td>
      <td>47</td>
      <td>4</td>
      <td>26</td>
      <td>115.9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>40</td>
      <td>23</td>
      <td>34</td>
      <td>83.8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>66</td>
      <td>9</td>
      <td>12</td>
      <td>113.3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10</td>
      <td>68</td>
      <td>8</td>
      <td>12</td>
      <td>109.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import StandardScaler

data_x = df.values[:, :4]
data_y = df.values[:, -1]

# 标准化数据
std_data = StandardScaler().fit_transform(df.values)
std_x = std_data[:, :4]
std_y = std_data[:, -1]

# 相关系数矩阵
r = np.corrcoef(data_x.T)
print(r)
```

    [[ 1.          0.22857947 -0.82413376 -0.24544511]
     [ 0.22857947  1.         -0.13924238 -0.972955  ]
     [-0.82413376 -0.13924238  1.          0.029537  ]
     [-0.24544511 -0.972955    0.029537    1.        ]]



```python
# 得到相关系数矩阵特征值与特征向量
eigenvalue, featurevector = np.linalg.eig(r)
print(eigenvalue)

# 使得每一个特征值的所有分量和为正
# 注意，每一列是特征向量而非行
f = np.sign(np.sum(featurevector, axis=0))
featurevector = f.reshape(1, -1)*featurevector
```

    [2.23570403e+00 1.57606607e+00 1.86606149e-01 1.62374573e-03]


特征值分别为 2.235,1.576,0.187,0.0016，略去第 4 个主成分，保留前三个特征值对应的三个特征向量，并计算前三个主成分


```python
# 前三个特征向量，即主成分系数
c = featurevector[:, :3]
# 降维的数据
data_xx = std_x.dot(c)
```

使用原始数据和降维后的数据做回归分析，并进行比较


```python
from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression as LR

# 定义误差函数
def residuals(p, x, y):
    return p[0] + x.dot(p[1:]) - y

# 计算原始数据回归方程系数
para1 = np.zeros(5)
r1 = leastsq(residuals, para1, (data_x, data_y))
# 计算主成分回归方程系数
para2 = np.zeros(4)
r2 = leastsq(residuals, para2, (data_xx, std_y))
r22 = c.dot(r2[0][1:])
r23 = r22 * data_y.std()/data_x.std(axis=0)
r23 = np.insert(r23, 0, data_y.mean() - data_x.mean(axis=0).dot(r23))

# # 另外一种回归方法
# r1 = LR().fit(data_x, data_y)
# print(r1.intercept_, r1.coef_)
# r2 = LR().fit(data_xx, data_y)
# print(r2.intercept_, r2.coef_)

# 打印
print("原始数据回归方程系数为", r1[0])
print("主成分回归方程系数为  ", r2[0])
print("标准化变量方程系数为  ", r22)
print("转化成原始自变量系数为", r23)
```

    原始数据回归方程系数为 [62.40532151  1.55110305  0.51016809  0.10190982 -0.14406053]
    主成分回归方程系数为   [-2.33405018e-10  6.56958050e-01  8.30863185e-03  3.02770243e-01]
    标准化变量方程系数为   [ 0.51297502  0.27868115 -0.06078483 -0.42288461]
    转化成原始自变量系数为 [85.74326351  1.3118899   0.26941931 -0.14276536 -0.3800747 ]



```python

```
