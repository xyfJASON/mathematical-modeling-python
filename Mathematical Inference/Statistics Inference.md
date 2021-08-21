<h1 style="text-align:center"> 统计推断 </h1>
<h2 style="text-align:center"> Statistics Inference </h2>
<div style="text-align:center"> xyfJASON </div>



参数估计和假设检验都是统计推断（Statistics Inference）的内容，相关参考资料可见：

- Introduction to Probability 《概率导论》第八章和第九章
- Statistics Inference 《统计推断》

以下内容仅为最简要的概述。

> 本文记号遵循《概率导论》，具体如下：
>
> - $\Theta$ 表示参数，$X$ 表示观测量，$x$ 表示观测值
> - $p_X(x)$ 表示离散随机变量 $X$ 的分布列，$f_X(x)$ 表示连续随机变量 $X$ 的概率密度函数
> - $P(A|B)$ 表示条件概率，$p_{X|Y}(x|y)$ 或 $f_{X|Y}(x|y)$ 表示条件分布列/条件概率密度函数
> - $\mathbb E[X|Y=y]$ 表示 $Y=y$ 的条件下 $X$ 的期望

贝叶斯学派将待未知模型或变量视为已知分布的随机变量，首先构造一个先验概率分布 $p_\Theta(\theta)$，在已知数据 $x$ 的情况下，使用贝叶斯公式推导后验概率分布 $p_{\Theta|X}(\theta|x)$，如此抓住 $x$ 提供的关于 $\theta$ 的所有信息。

经典学派将未知参数视为常数，并且对其进行估计。经典方法处理多个待选的概率模型，每个标记为 $\theta$ 的一个可能值。



## 1 贝叶斯统计推断



### 1.1 参数估计

首先确定一个待估计参数的先验概率分布，然后给出观测值 $x$，利用贝叶斯准则计算出后验概率分布 $p_{\Theta|X}(\theta|x)$（$\Theta$ 离散）或 $f_{\Theta|X}(\theta|x)$（$\Theta$ 连续），接下来有两种处理方式：

- 最大后验概率准则（MAP）：选取
  $$
  \hat\theta=\begin{cases}
  \arg\max\limits_\theta p_{\Theta|X}(\theta|x)&\Theta\text{ 离散}\\
  \arg\max\limits_\theta f_{\Theta|X}(\theta|x)&\Theta\text{ 连续}
  \end{cases}
  \quad=\quad\begin{cases}
  \arg\max\limits_\theta p_\Theta(\theta)p_{X|\Theta}(x|\theta)&\Theta\text{ 离散，}X\text{ 离散}\\
  \arg\max\limits_\theta p_\Theta(\theta)f_{X|\Theta}(x|\theta)&\Theta\text{ 离散，}X\text{ 连续}\\
  \arg\max\limits_\theta f_\Theta(\theta)p_{X|\Theta}(x|\theta)&\Theta\text{ 连续，}X\text{ 离散}\\
  \arg\max\limits_\theta f_\Theta(\theta)f_{X|\Theta}(x|\theta)&\Theta\text{ 连续，}X\text{ 连续}\\
  \end{cases}
  $$
  作为参数的估计值。

- 最小均方估计（LMS）：选取
  $$
  \hat\theta=\mathbb E[\Theta|X=x]=\begin{cases}\sum\limits_\theta p_{\Theta|X}(\theta|x)&\Theta\text{ 离散}\\\int\limits_\theta f_{\Theta|X}(\theta|x)\mathrm d\theta&\Theta\text{ 连续}\end{cases}
  $$
  作为参数的估计值。

  最小均方估计的特点是其使得估计量的均方误差最小。



### 1.2 假设检验

在一个假设检验问题中，$\Theta$ 取 $\theta_1,\ldots,\theta_m$ 中的**一个值**，其中 $m$ 是一个较小的整数。$m=2$ 成为二重假设检验问题。称事件 $\{\Theta=\theta_i\}$ 为第 $i$ 个假设，记为 $H_i$。

使用**最大后验概率准则**作假设检验，即选取使得后验概率
$$
P(\Theta=\theta_i|X=x)
$$
最大的假设 $H_i$，或等价地，选取使得
$$
\begin{cases}
p_\Theta(\theta_i)p_{X|\Theta}(x|\theta_i)&X\text{ 离散}\\
p_\Theta(\theta_i)f_{X|\Theta}(x|\theta_i)&X\text{ 连续}
\end{cases}
$$
最大的假设 $H_i$。



---



## 2 经典统计推断



### 2.1 参数估计——最大似然估计

设观测向量 $X=(X_1,\ldots,X_n)$ 的联合分布列/联合概率密度函数为 $p_X(x;\theta)$/$f_X(x;\theta)$，其中 $x=(x_1,\ldots,x_n)$ 为 $X$ 的观测值，那么最大似然估计指：
$$
\hat\theta=\begin{cases}
\arg\max\limits_\theta p_X(x_1,\ldots,x_n;\theta)&X\text {离散}\\
\arg\max\limits_\theta f_X(x_1,\ldots,x_n;\theta)&X\text {连续}
\end{cases}
$$
称 $p_X(x;\theta)$ 或 $f_X(x;\theta)$ 为似然函数，取对数后为对数似然函数。

对比贝叶斯参数估计的 MAP 准则，可以看到最大似然估计相当于贝叶斯估计中先验分布取均匀分布的情况。



### 2.2 二重假设检验——似然比检验

二重假设检验问题只考虑两个假设：原假设 $H_0$ 和备择假设 $H_1$。$H_0$ 是默认的模型，我们需要根据得到的数据决定是否拒绝 $H_0$。

记 $p_X(x;H)$/$f_X(x;H)$ 为在假设 $H$ 下 $x$ 的分布列/概率密度函数，定义似然比：
$$
L(x)=\frac{p_X(x;H_1)}{p_X(x;H_0)}\quad\text{或}\quad L(x)=\frac{f_X(x;H_1)}{f_X(x;H_0)}
$$
设定临界值 $\xi$，那么当 $L(x)>\xi$ 时，我们就拒绝原假设 $H_0$。

$\xi$ 是人为设定的，可以发现，$\xi=1$ 正好对应了最大似然准则。**我们可以设定 $\xi$ 以使得错误拒绝的概率 $P(L(x)>\xi;H_0)$ 为常数 $\alpha$。**



### 2.3 显著性检验

当假设不止两个特定的选择时，似然比检验不再适用，这时简单原假设 $H_0$ 为断言 $\theta$ 等于某个给定的元素 $\theta^*$，备择假设 $H_1$ 为 $H_0$ 不正确，即 $\theta\neq \theta^*$。

在观测数据之前，完成以下步骤：

1. 选择统计量 $S=h(X_1,\ldots,X_n)$
2. 确定拒绝域的形状，此时涉及到一个尚未确定的常数临界值 $\xi$
3. 选择显著水平 $\alpha$，即错误拒绝的概率为 $\alpha$
4. 确定临界值 $\xi$，使得错误拒绝的概率近似或等于 $\alpha$，此时就确定下了拒绝域

观测到数据 $x_1,\ldots,x_n$ 后，计算统计量 $S$ 的值 $s=h(x_1,\ldots,x_n)$，若 $s$ 落入拒绝域，则拒绝假设 $H_0$。



### 2.4 拟合优度检验

检验给定的分布列是否和观测数据保持一致。

**广义似然比检验**：

1. 通过最大似然来估计模型，得到使似然函数最大的参数向量 $\hat\theta$

2. 进行似然比检验，即计算广义似然比：
   $$
   \frac{p_X(x;\hat\theta)}{p_X(x;\theta^*)}
   $$
   其中 $\theta^*$ 是原假设 $H_0$ 的模型。若广义似然比超过临界值 $\xi$，则拒绝假设。同样我们可以选择 $\xi$ 以使得错误拒绝的概率近似或等于 $\alpha$。

<br>

**$\chi^2$ 检验**：设 $\theta^*_k$ 是假设 $H_0$ 下随机变量取值 $k$ 的概率，抽取样本量为 $n$ 的样本，令 $N_k$ 是样本中结果为 $k$ 的次数，观测值为 $n_k$。

- 利用统计量：
  $$
  S=\sum_{k=1}^mN_k\ln\left(\frac{N_k}{n\theta^*_k}\right)
  $$
  或其二阶泰勒展开的近似量的两倍：
  $$
  T=\sum_{k=1}^m\frac{(N_k-n\theta^*_k)^2}{n\theta^*_k}
  $$
  以及拒绝域：
  $$
  \{2S>\gamma\}\quad\text{或}\quad \{T>\gamma\}
  $$
  进行检验。

- 临界值 $\gamma$ 按照自由度为 $m-1$ 的 $\chi^2$ 分布的概率分布函数表确定，满足：
  $$
  P(2S>\gamma;H_0)=\alpha
  $$
  其中 $\alpha$ 是给定的显著性水平。


