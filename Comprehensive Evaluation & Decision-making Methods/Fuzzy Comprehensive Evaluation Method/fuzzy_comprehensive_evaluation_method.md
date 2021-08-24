# 模糊综合评价法
## 一级模糊在人事考核中的应用
1. 确定因素集

   对员工的表现需要从多个指标进行评判，所有这些指标构成了因素集，记
   $$U=\{u_1,u_2,\cdots,u_n\}$$
2. 确定评语集

   每个指标都有相关的评价值，如好、较好、差等，这些评价值构成了评语集，记
   $$V=\{v_1,v_2,\cdots,v_m\}$$
3. 确定各因素的权重

   员工不同表现的权重不同，可以通过 Delphi 法、加权平均法、众人评估法确定权重，记
   $$A=[a_1,a_2,\cdots,a_n]$$
4. 确定模糊综合判断矩阵

   对于指标 $u_i$，其下属的所有评价在评语集中所占的比例构成对指标 $u_i$ 的评判
   $$R_i=[r_{i1},r_{i2},\cdots,r_{im}$$
   各指标的模糊综合判断矩阵为
   $$R=\begin{bmatrix}
        r_{11} & r_{12} & \cdots & r_{1m}\\
        r_{21} & r_{22} & \cdots & r_{2m}\\
        \vdots & \vdots & \ddots & \vdots\\
        r_{n1} & r_{n2} & \cdots & r_{nm}\\
     \end{bmatrix}$$
5. 综合评判

   评判结果
   $$B=A\cdot R$$

某单位对员工进行年终综合评定：
1. 因素集 $U=\{政治表现u_1,工作能力u_2,工作态度u_3,工作成绩u_4\}$
2. 评语集 $V=\{优秀v_1,良好v_2,一般v_3,较差v_4,差v_5\}$
3. 各因素权重 $A=[0.25,0.2,0.25,0.3]$
4. 确定模糊综合评判矩阵

{u_1} 由群众打分，$R_1=[0.1,0.5,0.4,0,0]$，代表10%的人认为其政治表现优秀，50%的人认为政治表现良好，40%的人认为政治表现一般，其他为0。采用同样的方法评价其他因素，如$u_2,u_3$由领导部门打分，$u_4$ 由单位考核组成员打分确定，最后得到矩阵
$$R=\begin{bmatrix}
        0.1 & 0.5 & 0.4 & 0 & 0\\
        0.2 & 0.5 & 0.2 & 0.1 & 0\\
        0.2 & 0.5 & 0.3 & 0 & 0\\
        0.2 & 0.6 & 0.2 & 0 & 0\\
     \end{bmatrix}$$
5. 模糊综合评判

$$B=A\cdot R$$


```python
import numpy as np
A = np.array([0.25,0.2,0.25,0.3])
R = np.array([
     [0.1 , 0.5 , 0.4 , 0 , 0],
     [0.2 , 0.5 , 0.2 , 0.1,0],
     [0.2 , 0.5 , 0.3 , 0 , 0],
     [0.2 , 0.6 , 0.2 , 0 , 0],
])
B = A.dot(R)
B
```




    array([0.175, 0.53 , 0.275, 0.02 , 0.   ])



选取数值最大的评语作为综合评判结果，评判结果为'良好'

## 多层次模糊综合评判在人事考核中的应用
当评判指标过多时，可以应用多层次模糊综合评判进行评价。首先将因素集 $U$ 按照某种属性划分成 $s$ 个子因素集 $U_1,U_2,\cdots,U_s$，分别对 $U_i$ 进行一级模糊评判，再将评判结果看做一个整体，对所有子因素集，再次进行一级模糊评判。


```python
jt -t gruvboxl -f roboto -fs 12 -cellw 100% -T -N
```


      File "<ipython-input-20-97282b9c8444>", line 1
        jt -t gruvboxl -f roboto -fs 12 -cellw 100% -T -N
              ^
    SyntaxError: invalid syntax
    



```python
import pandas as pd
df = pd.read_csv('./data/mhdata.txt', sep='\t', header=None)
df.columns = [['评价', '评价', '评价','评价','评价'], ['优秀', '良好', '一般', '较差', '差']]
df.index = [['工作绩效','工作绩效','工作绩效','工作绩效', '工作态度', '工作态度', '工作态度', '工作态度', '工作态度', '工作能力', '工作能力', '工作能力', '工作能力', '工作能力', '学习特长', '学习特长', '学习特长', '学习特长'], ['工作量', '工作效率', '工作质量', '计划性', '责任感', '团队精神', '学习态度', '工作主动性',
           '满意度', '创新能力', '自我管理能力', '沟通能力', '协调能力', '执行能力', '勤情评价', '技能提高', '培训参与', '工作提案']]
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
      <th></th>
      <th colspan="5" halign="left">评价</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>优秀</th>
      <th>良好</th>
      <th>一般</th>
      <th>较差</th>
      <th>差</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">工作绩效</th>
      <th>工作量</th>
      <td>0.8</td>
      <td>0.15</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>工作效率</th>
      <td>0.2</td>
      <td>0.60</td>
      <td>0.10</td>
      <td>0.10</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>工作质量</th>
      <td>0.5</td>
      <td>0.40</td>
      <td>0.10</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>计划性</th>
      <td>0.1</td>
      <td>0.30</td>
      <td>0.50</td>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">工作态度</th>
      <th>责任感</th>
      <td>0.3</td>
      <td>0.50</td>
      <td>0.15</td>
      <td>0.05</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>团队精神</th>
      <td>0.2</td>
      <td>0.20</td>
      <td>0.40</td>
      <td>0.10</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>学习态度</th>
      <td>0.4</td>
      <td>0.40</td>
      <td>0.10</td>
      <td>0.10</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>工作主动性</th>
      <td>0.1</td>
      <td>0.30</td>
      <td>0.30</td>
      <td>0.20</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>满意度</th>
      <td>0.3</td>
      <td>0.20</td>
      <td>0.20</td>
      <td>0.20</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">工作能力</th>
      <th>创新能力</th>
      <td>0.1</td>
      <td>0.30</td>
      <td>0.50</td>
      <td>0.10</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>自我管理能力</th>
      <td>0.2</td>
      <td>0.30</td>
      <td>0.30</td>
      <td>0.10</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>沟通能力</th>
      <td>0.2</td>
      <td>0.30</td>
      <td>0.35</td>
      <td>0.15</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>协调能力</th>
      <td>0.1</td>
      <td>0.30</td>
      <td>0.40</td>
      <td>0.10</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>执行能力</th>
      <td>0.1</td>
      <td>0.40</td>
      <td>0.30</td>
      <td>0.10</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">学习特长</th>
      <th>勤情评价</th>
      <td>0.3</td>
      <td>0.40</td>
      <td>0.20</td>
      <td>0.10</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>技能提高</th>
      <td>0.1</td>
      <td>0.40</td>
      <td>0.30</td>
      <td>0.10</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>培训参与</th>
      <td>0.2</td>
      <td>0.30</td>
      <td>0.40</td>
      <td>0.10</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>工作提案</th>
      <td>0.4</td>
      <td>0.30</td>
      <td>0.20</td>
      <td>0.10</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



设置一级指标权重以及二级指标权重


```python
A =  np.array([0.4, 0.3, 0.2, 0.1])
A1 = np.array([0.2, 0.3, 0.3, 0.2])
A2 = np.array([0.3, 0.2, 0.1, 0.2, 0.2])
A3 = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
A4 = np.array([0.3, 0.2, 0.2, 0.3])
```

对各个子因素集进行一级模糊综合评判


```python
# 首先获得综合评判矩阵
R1 = df.values[:4, :]
R2 = df.values[4:9, :]
R3 = df.values[9:14, :]
R4 = df.values[14:, :]

# 得到评判结果
B1 = A1.dot(R1)
B2 = A2.dot(R2)
B3 = A3.dot(R3)
B4 = A4.dot(R4)
```

将各属性评判结果作为指标，再次进行一级模糊综合评价


```python
R = np.stack((B1, B2, B3, B4), axis=0)
B = A.dot(R)
B
```




    array([0.288 , 0.354 , 0.2355, 0.0865, 0.036 ])



选取最大值所对应的评价，可以认为对该员工的评价为良好。
