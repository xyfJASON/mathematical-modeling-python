# 灰色关联分析法
## 具体步骤
1. 确定比较对象（评价对象）和参考数列（评价标准）

设有 $m$ 个评价对象，$n$ 个评价指标，则参考数列为 $x_0=\{x_0(k)|k=1,2,\cdots,n\}$，比较数列为 $x_i=\{x_i(k)|k=1,2,\cdots,n\},i=1,2,\cdots,m$

2. 确定各指标值对应的权重

$w=[w_1,\cdots,w_n]$

3. 计算灰色关联系数
$$
\xi_i(k)=\frac{\min\limits_s\min\limits_t|x_0(t)-x_s(t)|+\rho\max\limits_s\max\limits_t|x_0(t)-x_s(t)|}
{|x_0(k)-x_i(k)|+\rho\max\limits_s\max\limits_t|x_0(t)-x_s(t)|}
$$
称 'minmin' 项为两级最小差，'maxmax' 项为两级最大差，$\rho$ 为分辨系数，$\rho$ 越大，分辨率越大

4. 计算灰色加权关联度。其计算公式为
$$
r_i=\mathop{\Sigma}\limits_{k=1}^nw_i\xi_i(k)
$$
$r_i$ 为第 $i$ 个评价对象对理想对象的灰色加权关联度。

5. 评价分析

根据灰色加权关联度的大小，对各评价对象进行排序，可以建立评价对象的关联序，关联度越大，评价结果越好。

## 实例
在 6 个待选的零部件供应商中选择一个合作伙伴，各待选供应商有关数据如下


```python
import pandas as pd
import numpy as np

df = pd.DataFrame([[ 0.83 , 0.9  , 0.99 , 0.92 , 0.87 , 0.95 ],
     [ 326  , 295  , 340  , 287  , 310  , 303  ],
     [ 21   , 38   , 25   , 19   , 27   , 10   ],
     [ 3.2  , 2.4  , 2.2  , 2    , 0.9  , 1.7  ],
     [ 0.2  , 0.25 , 0.12 , 0.33 , 0.2  , 0.09 ],
     [ 0.15 , 0.2  , 0.14 , 0.09 , 0.15 , 0.17 ],
     [ 250  , 180  , 300  , 200  , 150  , 175  ],
     [ 0.23 , 0.15 , 0.27 , 0.3  , 0.18 , 0.26 ],
     [ 0.87 , 0.95 , 0.99 , 0.89 , 0.82 , 0.94 ]])
df.columns = [['待选供应商']*6, list('123456')]
df.index = ['产品质量', '产品价格', '地理位置', '售后服务', '技术水平', '经济效益', '供应能力', '市场影响度', '交货情况']
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

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="6" halign="left">待选供应商</th>
    </tr>
    <tr>
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>产品质量</th>
      <td>0.83</td>
      <td>0.90</td>
      <td>0.99</td>
      <td>0.92</td>
      <td>0.87</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>产品价格</th>
      <td>326.00</td>
      <td>295.00</td>
      <td>340.00</td>
      <td>287.00</td>
      <td>310.00</td>
      <td>303.00</td>
    </tr>
    <tr>
      <th>地理位置</th>
      <td>21.00</td>
      <td>38.00</td>
      <td>25.00</td>
      <td>19.00</td>
      <td>27.00</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>售后服务</th>
      <td>3.20</td>
      <td>2.40</td>
      <td>2.20</td>
      <td>2.00</td>
      <td>0.90</td>
      <td>1.70</td>
    </tr>
    <tr>
      <th>技术水平</th>
      <td>0.20</td>
      <td>0.25</td>
      <td>0.12</td>
      <td>0.33</td>
      <td>0.20</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>经济效益</th>
      <td>0.15</td>
      <td>0.20</td>
      <td>0.14</td>
      <td>0.09</td>
      <td>0.15</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>供应能力</th>
      <td>250.00</td>
      <td>180.00</td>
      <td>300.00</td>
      <td>200.00</td>
      <td>150.00</td>
      <td>175.00</td>
    </tr>
    <tr>
      <th>市场影响度</th>
      <td>0.23</td>
      <td>0.15</td>
      <td>0.27</td>
      <td>0.30</td>
      <td>0.18</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>交货情况</th>
      <td>0.87</td>
      <td>0.95</td>
      <td>0.99</td>
      <td>0.89</td>
      <td>0.82</td>
      <td>0.94</td>
    </tr>
  </tbody>
</table>
</div>



其中产品质量、技术水平、供应能力、经济效益、交货情况、市场影响度指标为效益型指标，其标准化方式为
$$
std = (ori - min(ori))/(max(ori) - min(ori))
$$
而产品地位、地理位置、售后服务指标属于成本型指标，其标准化方式为
$$
std = (max(ori) - ori))/(max(ori) - min(ori))
$$


```python
df = df.T
# 对数据进行预处理
df1 = df.iloc[:, [0, 4, 5, 6, 7, 8]]
df1 = (df1 - df1.min())/(df1.max() - df1.min())
df2 = df.iloc[:, [1, 2, 3]]
df2 = (df2.max() - df2)/(df2.max() - df2.min())
df.iloc[:, [0, 4, 5, 6, 7, 8]] = df1
df.iloc[:, [1, 2, 3]] = df2
df = df.T
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

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="6" halign="left">待选供应商</th>
    </tr>
    <tr>
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>产品质量</th>
      <td>0.000000</td>
      <td>0.437500</td>
      <td>1.000000</td>
      <td>0.562500</td>
      <td>0.250000</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>产品价格</th>
      <td>0.264151</td>
      <td>0.849057</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.566038</td>
      <td>0.698113</td>
    </tr>
    <tr>
      <th>地理位置</th>
      <td>0.607143</td>
      <td>0.000000</td>
      <td>0.464286</td>
      <td>0.678571</td>
      <td>0.392857</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>售后服务</th>
      <td>0.000000</td>
      <td>0.347826</td>
      <td>0.434783</td>
      <td>0.521739</td>
      <td>1.000000</td>
      <td>0.652174</td>
    </tr>
    <tr>
      <th>技术水平</th>
      <td>0.458333</td>
      <td>0.666667</td>
      <td>0.125000</td>
      <td>1.000000</td>
      <td>0.458333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>经济效益</th>
      <td>0.545455</td>
      <td>1.000000</td>
      <td>0.454545</td>
      <td>0.000000</td>
      <td>0.545455</td>
      <td>0.727273</td>
    </tr>
    <tr>
      <th>供应能力</th>
      <td>0.666667</td>
      <td>0.200000</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>市场影响度</th>
      <td>0.533333</td>
      <td>0.000000</td>
      <td>0.800000</td>
      <td>1.000000</td>
      <td>0.200000</td>
      <td>0.733333</td>
    </tr>
    <tr>
      <th>交货情况</th>
      <td>0.294118</td>
      <td>0.764706</td>
      <td>1.000000</td>
      <td>0.411765</td>
      <td>0.000000</td>
      <td>0.705882</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 获取比较对象与参考数列
x = df.values
x0 = x.max(axis=1)

# 计算灰色关联系数
def grey_degree(x, x0, rho):
    x0 = x0.reshape(-1, 1)
    a = np.fabs(x0 - x)
    b = np.min(a)
    c = np.max(a)
    return (b + rho*c)/(a + rho*c)

index = grey_degree(x, x0, 0.5)
index
```




    array([[0.33333333, 0.47058824, 1.        , 0.53333333, 0.4       ,
            0.66666667],
           [0.40458015, 0.76811594, 0.33333333, 1.        , 0.53535354,
            0.62352941],
           [0.56      , 0.33333333, 0.48275862, 0.60869565, 0.4516129 ,
            1.        ],
           [0.33333333, 0.43396226, 0.46938776, 0.51111111, 1.        ,
            0.58974359],
           [0.48      , 0.6       , 0.36363636, 1.        , 0.48      ,
            0.33333333],
           [0.52380952, 1.        , 0.47826087, 0.33333333, 0.52380952,
            0.64705882],
           [0.6       , 0.38461538, 1.        , 0.42857143, 0.33333333,
            0.375     ],
           [0.51724138, 0.33333333, 0.71428571, 1.        , 0.38461538,
            0.65217391],
           [0.41463415, 0.68      , 1.        , 0.45945946, 0.33333333,
            0.62962963]])




```python
# 假设各项指标地位相同，权重相等
# 计算关联度
r = index.mean(axis=0)
r

# 列表
df2 = pd.DataFrame(np.concatenate((index, r.reshape(1, -1)), axis=0))
df2.columns = [f'供应商{i}' for i in '123456']
df2.index = [*[f'指标{i}' for i in range(1, 10)], 'r']
df2
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
      <th>供应商1</th>
      <th>供应商2</th>
      <th>供应商3</th>
      <th>供应商4</th>
      <th>供应商5</th>
      <th>供应商6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>指标1</th>
      <td>0.333333</td>
      <td>0.470588</td>
      <td>1.000000</td>
      <td>0.533333</td>
      <td>0.400000</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>指标2</th>
      <td>0.404580</td>
      <td>0.768116</td>
      <td>0.333333</td>
      <td>1.000000</td>
      <td>0.535354</td>
      <td>0.623529</td>
    </tr>
    <tr>
      <th>指标3</th>
      <td>0.560000</td>
      <td>0.333333</td>
      <td>0.482759</td>
      <td>0.608696</td>
      <td>0.451613</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>指标4</th>
      <td>0.333333</td>
      <td>0.433962</td>
      <td>0.469388</td>
      <td>0.511111</td>
      <td>1.000000</td>
      <td>0.589744</td>
    </tr>
    <tr>
      <th>指标5</th>
      <td>0.480000</td>
      <td>0.600000</td>
      <td>0.363636</td>
      <td>1.000000</td>
      <td>0.480000</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>指标6</th>
      <td>0.523810</td>
      <td>1.000000</td>
      <td>0.478261</td>
      <td>0.333333</td>
      <td>0.523810</td>
      <td>0.647059</td>
    </tr>
    <tr>
      <th>指标7</th>
      <td>0.600000</td>
      <td>0.384615</td>
      <td>1.000000</td>
      <td>0.428571</td>
      <td>0.333333</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>指标8</th>
      <td>0.517241</td>
      <td>0.333333</td>
      <td>0.714286</td>
      <td>1.000000</td>
      <td>0.384615</td>
      <td>0.652174</td>
    </tr>
    <tr>
      <th>指标9</th>
      <td>0.414634</td>
      <td>0.680000</td>
      <td>1.000000</td>
      <td>0.459459</td>
      <td>0.333333</td>
      <td>0.629630</td>
    </tr>
    <tr>
      <th>r</th>
      <td>0.462992</td>
      <td>0.555994</td>
      <td>0.649074</td>
      <td>0.652723</td>
      <td>0.493562</td>
      <td>0.613015</td>
    </tr>
  </tbody>
</table>
</div>


