<h1 style="text-align:center"> 层次分析法 </h1>
<h2 style="text-align:center"> Analytic Hierarchy Process </h2>
<div style="text-align:center"> xyfJASON </div>



## 1 理论

在许多问题中，我们会面临若干选项进行选择，层次分析法（AHP）可以帮助我们对一些较为复杂、模糊的问题做出决策。层次分析法是一个主观性较强的方法，特别适用于难以定量分析的问题。

层次分析法的步骤如下：

1. 建立层次结构
2. 各层次建立判别矩阵
3. 层次单排序、一致性检验
4. 层次总排序、一致性检验



### 1.1 建立层次结构

一个复杂问题被分解为若干元素，元素构成一种层次的结构，前一层影响后一层。例如：

<img src="img/AHP.png" style="zoom: 33%;" />

目标层只有一个元素，一般是问题的目标或结果；

准则层可以有多个层次，主要是一些指标和准则；

措施层包含可供选择的若干选项。



### 1.2 各层次建立判别矩阵

对于某一层而言，假设该层有 $n$ 个元素，对于更高一层的某一个元素 $Z$，建立判别矩阵 $A=(a_{ij})_{n\times n}$，满足：

1. $a_{ij}>0$ 且 $a_{ji}=\frac{1}{a_{ij}}$

2. $a_{ij}$ 的含义为：

   <img src="img/AHP2.png" style="zoom: 50%;" />



### 1.3 层次单排序、一致性检验

取出判别矩阵 $A$ 的最大特征值 $\lambda_\max$ 对应的特征向量 $W$，归一化后即得到该层各元素相对于高一层元素 $Z$ 的权重。

一致性：$a_{ij}a_{jk}=a_{ik}$

由于判别矩阵是人为构造的，一般不会满足一致性，因而我们需要做一致性检验，只要不差得太离谱即可。

一致性检验步骤如下：

1. 计算一致性指标
   $$
   CI=\frac{\lambda_\max-n}{n-1}
   $$

2. 查找相应的平均随机一致性指标

   | $n$  |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   |  10  |
   | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
   | $RI$ |  0   |  0   | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 | 1.49 |

3. 计算一致性比例
   $$
   CR=\frac{CI}{RI}
   $$
   若 $CR<0.10$，认为判别矩阵的一致性可以接受，否则需要对判别矩阵进行修正。



### 1.4 层次总排序、一致性检验

自上而下将计算总权重。具体地，设高一层的总权重已计算出来，为 $a_1,\ldots,a_m$，根据前一个步骤，已知当前层的每个元素对高一层每个元素都有一个权重 $b_{ij},\,i=1,\cdots,n,\,j=1,\ldots,m$，则当前层总权重为：
$$
b_i=\sum_{j=1}^ma_{j}b_{ij},\quad i=1,\ldots,n
$$
虽然每一层已经做过一致性检验了，但是不一致性可能在结果中叠加，因而还需作一致性检验。具体地，对于高一层的第 $j$ 个元素，在层次单排序时已经得到了当前层的一致性指标 $CI(j)$ 和相应平均随机一致性指标 $RI(j)$，则当前层一致性比例为：
$$
CR=\frac{\sum\limits_{j=1}^ma_j\cdot CI(j)}{\sum\limits_{j=1}^ma_j\cdot RI(j)}
$$
若 $CR<0.10$，接受一致性。



---



## 2 代码模板

```python
class AHP:
    def __init__(self,
                 sz_layers: list[int],
                 judge_mat: list[list[np.ndarray]]) -> None:
        """
        :param sz_layers: number of elements in each layer
        :param judge_mat: judgement matrices
        """
        self.n_layers = len(sz_layers)
        assert len(judge_mat) == self.n_layers - 1
        for i in range(self.n_layers - 1):
            assert np.stack(judge_mat[i], axis=0).shape == (sz_layers[i+1], sz_layers[i], sz_layers[i])
        for sz in sz_layers:
            assert 1 <= sz <= 10
        self.sz_layers = sz_layers
        self.judge_mat = judge_mat

        self.CI = []
        self.RI = []
        self.RI_table = [None, 0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
        self.b = []
        self.a = []

    def run(self) -> np.ndarray:
        for i in range(self.n_layers - 1):
            b, CI, RI = [], [], []
            for j in range(self.sz_layers[i+1]):
                mat = self.judge_mat[i][j]
                eigw, eigv = np.linalg.eig(mat)
                eigv, eigw = eigv[:, np.argmax(eigw)], np.max(eigw)
                eigv, eigw = eigv.real, eigw.real
                CI.append((eigw - self.sz_layers[i]) / (self.sz_layers[i] - 1))
                RI.append(self.RI_table[self.sz_layers[i]])
                assert CI[-1] / RI[-1] < 0.10, 'Cannot pass consistency test.'
                b.append(eigv / eigv.sum())
            self.b.append(b)
            self.CI.append(CI)
            self.RI.append(RI)
        self.a.append(np.array([1.0]))
        for i in range(self.n_layers - 2, -1, -1):
            a = []
            for j in range(self.sz_layers[i]):
                a.append(self.a[-1] @ np.vstack(self.b[i])[:, j])
            CR = (self.a[-1] @ np.array(self.CI[i])) / (self.a[-1] @ np.array(self.RI[i]))
            assert CR < 0.10, 'Cannot pass consistency test.'
            self.a.append(np.array(a))
        return self.a[-1]
```



---



## 3 例题



### 3.1 例一

经双方恳谈，已有三个单位表示愿意录用某毕业生。该生根据已有信息建立了一个层次结构模型，模型结构和相应判别矩阵如图所示：

<img src="img/AHP3.png" style="zoom:67%;" />

编写代码如下：

```python
CB1 = np.array([[1, 1/4, 1/2], [4, 1, 3], [2, 1/3, 1]])
CB2 = np.array([[1, 1/4, 1/5], [4, 1, 1/2], [5, 2, 1]])
CB3 = np.array([[1, 3, 1/3], [1/3, 1, 1/7], [3, 7, 1]])
CB4 = np.array([[1, 1/3, 5], [3, 1, 7], [1/5, 1/7, 1]])
CB5 = np.array([[1, 1, 7], [1, 1, 7], [1/7, 1/7, 1]])
CB6 = np.array([[1, 7, 9], [1/7, 1, 1], [1/9, 1, 1]])
BA = np.array([[1, 1, 1, 4, 1, 1/2],
               [1, 1, 2, 4, 1, 1/2],
               [1, 1/2, 1, 5, 3, 1/2],
               [1/4, 1/4, 1/5, 1, 1/3, 1/3],
               [1, 1, 1/3, 3, 1, 1],
               [2, 2, 2, 3, 3, 1]])
judge_mat = [[CB1, CB2, CB3, CB4, CB5, CB6], [BA]]

solver = AHP([3, 6, 1], judge_mat)
res = solver.run()
print(res)
```

结果如下：

```
[0.3951982  0.29962129 0.30518051]
```

由于第 1 个权重最大，故选择工作 1。

