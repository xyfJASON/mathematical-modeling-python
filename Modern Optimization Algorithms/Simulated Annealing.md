<h1 style="text-align:center"> 模拟退火 </h1>
<h2 style="text-align:center"> Simulated Annealing </h2>
<div style="text-align:center"> xyfJASON </div>



## 1 概述

定义状态 $i$ 的能量为 $E(i)$，在温度 $T$ 下，从状态 $i$ 进入状态 $j$ 遵循如下规律：

- 若 $E(j)\leqslant E(i)$​，则接受该转换；

- 否则，以如下概率接受转换：
  $$
  e^\cfrac{E(i)-E(j)}{KT}
  $$
  其中，$K$ 在物理意义上是玻尔兹曼常数，但在算法实现中我们取 $K=1$ 即可。

在**特定温度**下，充分次转换后状态的概率分布达到平衡（玻尔兹曼分布）：
$$
P_T(X=i)=\frac{\exp\left(-\frac{E(i)}{KT}\right)}{\sum\limits_{j\in S}\exp\left(-\frac{E(j)}{KT}\right)}
$$
容易知道，当 $T$ 很大时，该分布区倾向于均匀分布，即每种状态等可能出现；而当 $T$ 很小时，该分布倾向于集中在能量最低的状态上，即有很大概率进入能量最小状态。

综上，模拟退火算法步骤如下：

1. 设定初始温度、结束温度、降温比率、能量函数
2. 设定初始状态和状态转换规则，要求当前状态只依赖于前一个状态
3. 在当前温度下按照接受概率进行若干次状态转换
4. 降温
5. 重复 3、4 步骤直至温度低于结束温度

**因此，要用模拟退火算法解决一个实际问题，最关键的步骤是设计好状态表示、能量函数和状态转换规则。**



---



## 2 代码模板

```python
class SimulatedAnnealing:
    """
    To run simulated annealing, inherit this class, then
    override 'gen_init_state_energy` and `next_state_energy`.
    """
    def __init__(self,
                 init_T: float,
                 end_T: float,
                 cool_factor: float,
                 steps_per_T: int = 1) -> None:
        self.init_T = init_T
        self.end_T = end_T
        self.cool_factor = cool_factor
        self.steps_per_T = steps_per_T
        self.init_state, self.init_energy = self.gen_init_state_energy()
        self.record_energy = ([], [])  # 0: history; 1: best

    def reset(self) -> None:
        self.init_state, self.init_energy = self.gen_init_state_energy()
        self.record_energy = ([], [])

    def gen_init_state_energy(self) -> Tuple[object, float]:
        raise NotImplementedError

    def next_state_energy(self, cur_state: object, cur_energy: float) -> Tuple[object, float]:
        raise NotImplementedError

    def step(self, cur_T: float, cur_state: object, cur_energy: float) -> Tuple[object, float]:
        next_state, next_energy = self.next_state_energy(cur_state, cur_energy)
        delta_energy = max(0.0, next_energy - cur_energy)
        prob = np.exp(-delta_energy / cur_T)
        if np.random.rand() <= prob:
            return next_state, next_energy
        else:
            return cur_state, cur_energy

    def run(self) -> Tuple[object, float]:
        cur_T = self.init_T
        cur_state = self.init_state
        cur_energy = self.init_energy
        best_state = self.init_state
        best_energy = self.init_energy
        while cur_T > self.end_T:
            for _ in range(self.steps_per_T):
                cur_state, cur_energy = self.step(cur_T, cur_state, cur_energy)
                if best_energy > cur_energy:
                    best_state, best_energy = cur_state, cur_energy
                self.record_energy[0].append(cur_energy)
                self.record_energy[1].append(best_energy)
            cur_T *= self.cool_factor
        return best_state, best_energy

    def plot(self, history: bool = True, best: bool = True) -> None:
        assert history or best
        fig, ax = plt.subplots(1, 1)
        length = len(self.record_energy[0])
        if history:
            ax.plot(range(length), self.record_energy[0], label='history', c='dodgerblue')
        if best:
            ax.plot(range(length), self.record_energy[1], label='best', c='darkorange')
        ax.set_title('Energy curve')
        ax.set_xlabel('Iters')
        ax.set_ylabel('Energy')
        plt.legend()
        plt.show()
```



---



## 3 例题



### 3.1 例一——旅行商问题（TSP）

http://www.math.uwaterloo.ca/tsp/vlsi/index.html

模拟退火算法解 TSP 问题：

- 状态表示：城市编号序列，依次按照序列顺序访问城市
- 能量函数：路径长度
- 状态转换规则：在当前序列中随机选择一段连续自序列，翻转并插入回原位置

代码如下：

```python
def TSP():
    """
    For this problem, the best result is 564
    refer to https://www.math.uwaterloo.ca/tsp/vlsi/xqf131.tour.html
    """
    with open('tsp_data.txt') as f:
        data = [list(map(int, line.strip().split())) for line in f.readlines()]
        data = np.array(data)
    n = 131

    def distance(_i, _j):
        """ distance between the ith row and the jth row """
        return np.sqrt((data[_i, 1] - data[_j, 1]) ** 2 + (data[_i, 2] - data[_j, 2]) ** 2)

    class Solver(SimulatedAnnealing):
        def gen_init_state_energy(self) -> Tuple[np.ndarray, float]:
            best_init_state, best_init_energy = np.arange(n), np.inf
            for _ in range(10):
                init_state = np.random.permutation(n)
                init_energy = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    init_energy += distance(init_state[i], init_state[j])
                if init_energy < best_init_energy:
                    best_init_state, best_init_energy = init_state, init_energy
            return best_init_state, best_init_energy

        def next_state_energy(self, cur_state: np.ndarray, cur_energy: float) -> Tuple[np.ndarray, float]:
            u, v = np.random.choice(n+1, 2, replace=False)
            u, v = (v, u) if u > v else (u, v)
            next_state = cur_state.copy()
            next_state[u:v] = next_state[u:v][::-1]  # [u, v)
            next_energy = cur_energy
            if u > 0:
                next_energy -= distance(cur_state[u], cur_state[u-1])
                next_energy += distance(cur_state[v-1], cur_state[u-1])
            if v < n:
                next_energy -= distance(cur_state[v], cur_state[v-1])
                next_energy += distance(cur_state[v], cur_state[u])
            return next_state, next_energy

    solver = Solver(init_T=100,
                    end_T=1e-30,
                    cool_factor=0.999,
                    steps_per_T=1)
    res = solver.run()
    print(res)
    solver.plot()

    def plot_route(res_state):
        fig, ax = plt.subplots(1, 1)
        ax.scatter(data[:, 1], data[:, 2])
        ax.plot(np.hstack((data[res_state, 1], data[res_state[0:1], 1])),
                np.hstack((data[res_state, 2], data[res_state[0:1], 2])))
        plt.show()

    plot_route(res[0])
```

结果如下（每次运行结果不同）：

```python
(array([ 88,  92,  97, 111, 122, 129, 120, 117, 113, 104,  99, 100, 101,
       105, 106,  98,  93,  91,  87,  86,  81,  80,  77,  76,  74,  67,
        63,  73,  52,  44,  26,  25,  24,  17,  12,   4,  11,   0,   6,
         5,  13,  14,  15,  16,  18,  27,  45,  53,  54,  46,  47,  48,
        49,  50,  51,  56,  55,  61,  64,  68,  65,  69,  70,  66,  62,
        57,  37,  36,  35,  21,  34,  33,  32,  31,  30,  29,  28,  19,
        20,   7,   1,   2,   8,   3,  10,   9,  22,  38,  39,  40,  41,
        23,  42,  43,  60,  59,  58,  72,  71,  79,  75,  78,  82,  83,
        84,  85,  89,  90,  94,  95,  96, 103, 102, 110, 116, 109, 108,
       107, 112, 123, 124, 125, 126, 114, 118, 115, 119, 121, 128, 127,
       130]), 571.9007561620243)
```

![](img/sa_curve.png)

![](img/sa.png)



### 3.2 例二

求下列函数在 $[-3,3]^2$ 上的最大值：
$$
F(x,y)=3(1-x)^2e^{-x^2-(y+1)^2}-10\left(\frac{x}{5}-x^3-y^5\right)e^{-x^2-y^2}-\frac{1}{3}^{\exp\left(-(x+1)^2-y^2\right)}
$$
代码如下：

```python
def Ex():
    def F(x, y):
        return 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2)-10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)-1/3**np.exp(-(x+1)**2-y**2)

    class Solver(SimulatedAnnealing):
        def gen_init_state_energy(self) -> Tuple[np.ndarray, float]:
            return np.array([0, 0]), -F(0, 0)

        def next_state_energy(self, cur_state: np.ndarray, cur_energy: float) -> Tuple[np.ndarray, float]:
            next_state = cur_state + np.random.randn(2) / 10.
            next_state = np.clip(next_state, [-3, -3], [3, 3])
            return next_state, -F(next_state[0], next_state[1])

    solver = Solver(init_T=100,
                    end_T=1e-30,
                    cool_factor=0.996,
                    steps_per_T=1)
    res = solver.run()
    print(res[0], -res[1])
    solver.plot()
```

结果如下：

```
[0.10020633 1.62125861] 7.017346426963329
```

![](img/sa_curve2.png)

作图如下：

![](img/sa2.png)

