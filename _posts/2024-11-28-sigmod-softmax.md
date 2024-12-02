---
classes: wide2
title: "手撕 Softmax 和 Cross-entropy 公式及代码"
excerpt: "从公式以及代码的角度入手"
categories: 
  - coding
tags: 
  - contests
toc: true
toc_sticky: true
mathjax: true
---

# 前言

Softmax 通常用于将网络的输出控制到 [0,1] 范围内，而 Cross-entropy（交叉熵）通常用在分类任务，将模型的对 $k$ 个类别的预测结果与实际的标签之间计算出一个 loss，而这个 loss 通常使用交叉熵来实现。

> 注：本文假设读者有基础的机器学习知识。

# 理论

Softmax 本质上是把模型的输出做一个归一化，由于模型直接输出的数值有正有负，首先对所有 $k$ 个类别输出的结果做一个指数，得到 $k$ 个大于等于 0 的数，然后结果除以总和，使得最终的 $k$ 个数的和是 1。

## 正向传播

术语：

- $x$ 表示输入样本
- $h_i(x)$ 表示样本经过模型之后直接得到的对应于第 $i$ 个类别的值
- $k$ 表示总共有 $k$ 个类别
- $z_i = p(\text{label}=i)$ 表示模型预估输入样本为第 $i$ 个类别的概率

则有：

$z_i = p(\text{label}=i)=\dfrac{\exp(h_i(x))}{\sum_{j=1}^k\exp(h_j(x))}$ 

也就是

$z=\text{normalize}(\exp(h(x)))$ 

假设样本对应的正确标签为 $y$ ，定义交叉熵损失为

$\ell_{ce}(h(x),y)=-\log(p(\text{label}=y))$ 

此处的直观理解是对输出正确标签的概率取负对数，当输出正确标签的概率为 1 的时候，这个值为 0，表示误差最小，否则误差是正无穷。

然后我们把 Softmax 的那个定义（ $p(\text{label}=i)$ ）带入进去，可以得到：

$\ell_{ce}(h(x),y) = -\log(\dfrac{\exp(h_y(x))}{\sum_{j=1}^k\exp(h_j(x))}) =-h_y(x)+\log\sum_{j=1}^k\exp(h_j(x))$ 

因此单独算交叉熵损失的时候，Softmax 其实不用全部算出来。

## 反向传播

如果使用梯度下降算法对这个 loss 进行优化的话，需要求解模型参数的梯度，也就是对 $h$ 求导：

$$
\begin{align}
\dfrac{\partial \ell_{ce}(h,y)}{\partial h_i}=&\dfrac{\partial}{\partial h_i}(-h_y+\log\sum_{j=1}^k\exp(h_j))\\ 
=&-\mathbb{I}(i=y)+\dfrac{\dfrac{\partial}{\partial h_i}\sum_{j=1}^k\exp(h_j)}{\sum_{j=1}^k\exp(h_j)}\\ 
=&-\mathbb{I}(i=y)+\dfrac{\exp(h_i)}{\sum_{j=1}^k\exp(h_j)}\\ 
=&-\mathbb{I}(i=y)+z_i 
\end{align}
$$ 

对于梯度来说，则是一个向量：

$\nabla_h\ell_{ce}(h,y)=-e_y+z$ 

其中 $e_y$ 是一个长度为 $k$ 的单位向量，只在第 $y$ 个位置上是 1，其余地方是 0，这个时候我们可以发现，交叉熵的梯度实际上有一项直接就是 Softmax 的值。

## 小结

小结一下，目前我们推出来了两个式子：

$$
\ell_{ce}(h(x),y) = -h_y(x)+\log\sum_{j=1}^k\exp(h_j(x))
$$

$$
\nabla_h\ell_{ce}(h,y)=-e_y+z
$$

分别对应正向和反向传播。

## 线性模型

下面具体来看看 $h_\theta(x) = \theta^Tx$ ，也就是线性模型的场景。

此外我们额外考虑随机梯度下降的场景，也就是每次用 $B$ 个样本（mini-batch）做训练，并假设样本的特征维度为 $n$ ，即 $X\in\mathbb{R}^{B\times n}$ ，这 $B$ 个样本对应的标签为 $y\in\{1,...,k\}^B$ ，那么反向传播时使用以下公式进行梯度的更新：

$\theta:=\theta -\dfrac{\alpha}{B}\displaystyle\sum_{i=1}^B\nabla_\theta\ell(h_\theta(x^{(i)}),y^{(i)})$ 

其中 $\alpha$ 为学习率。

现在来求 $\nabla_\theta\ell_{ce}(\theta^Tx,y)$ ，首先观察一下维度：

- $x$ 维度为 $\mathbb{R}^{n}$ 
- $y$ 维度为 $\mathbb{R}$ 
- $\theta$ 维度为 $\mathbb{R}^{n\times k}$ 

期望得到的梯度 $\nabla_\theta$ 和 $\theta$ 具有相同的维度 $\mathbb{R}^{n\times k}$ 。

对于多元微积分，求导的正确做法其实是比较复杂的，但是实际上人们采用一种近似做法：首先把这些变量看成是标量，求完导之后，把变量的顺序调整一下使得最终的维度能够对齐...（最后通过数值计算来检验下结果）

比如对于这个求导，使用链式求导法则：

$$
\begin{align}
\dfrac{\partial}{\partial\theta}\ell_{ce}(\theta^Tx,y)=&\dfrac{\partial\ell_{ce}(\theta^T x,y)}{\partial(\theta^Tx)}\cdot\dfrac{\partial(\theta^Tx)}{\partial \theta}\\
=&(z-e_y)\cdot x
\end{align}
$$

其中最后一步的前半部分使用了之前的结论，然后对一下维度：

- $(z-e_y)\in\mathbb{R}^{k\times 1}$ 
- $x\in\mathbb{R}^{n\times1}$ 

为了得到 $\nabla_\theta\in\mathbb{R}^{n\times k}$ ，我们可以这样搞：

$\dfrac{\partial}{\partial\theta}\ell_{ce}(\theta^Tx,y)=x(z-e_y)^T$ 

然后再考虑带 batch 的情况，比如 batch size $B=m$ ，求 $\dfrac{\partial}{\partial\theta}\ell_{ce}(X\theta,y)$ ，其中 $X\in\mathbb{R}^{m\times n}$ 

那么有：

$$
\begin{align}
\dfrac{\partial}{\partial\theta}\ell_{ce}(X\theta,y) 
=&\dfrac{\partial\ell_{ce}(X\theta,y)}{\partial(X\theta)}\cdot\dfrac{\partial(X\theta)}{\partial \theta}\\ 
=&(Z-I_y)\cdot X
\end{align}
$$

要求的参数维度实际上是不变的：$\nabla_\theta\in\mathbb{R}^{n\times k}$

而

- $(Z-I_y)\in\mathbb{R}^{m\times k}$ 
- $X\in\mathbb{R}^{m\times n}$ 

为了凑出来维度，可以这样写：

$\dfrac{\partial}{\partial\theta}\ell_{ce}(X\theta,y)=X^T(Z-I_y)$ 

其中 $Z=\text{normalize}(\exp(X\theta))$ ，注意是按行做归一化。

# 代码

Softmax 实现如下：

其中给定 Z 为模型最终的预测结果，维度为 $m\times k$ ，也就是对每个样本有 $k$ 个值（取值范围为 $[0,1]$ ），分别表示模型对该样本预测为 $k$ 个类别的概率，y 则是真实标签，维度为 $m\times 1$ ，回忆一下，我们之前推出来

$$
\ell_{ce}(h(x),y) = -\log(\dfrac{\exp(h_y(x))}{\sum_{j=1}^k\exp(h_j(x))}) =-h_y(x)+\log\sum_{j=1}^k\exp(h_j(x))
$$

（我们需要把它对应到向量的写法，因为要写的是 $m$ 个样本对应得到的结果）

其中 $\log\sum_{j=1}^k\exp(h_j(x))$ 对应 `np.log(np.sum(mp.exp(Z), axis=1))`，其中 `axis=1` 保证了按行做求和，这部分对应的维度是 $m\times 1$ ，然后我们需要构造一个向量，维度为 $m\times 1$ ，其中的每个值是每个样本对应的 $-h_y(x)$ ，即 `-Z[np.arange(Z.shape[0]), y]`，两者相加然后求平均值即为最终结果 

```python3
def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    # \log(\sum(\exp(z))) - z
    return np.mean(np.log(np.sum(np.exp(Z), axis=1)) - Z[np.arange(Z.shape[0]), y])
```

线性模型手动计算梯度实现如下：

核心公式是

$$
\theta:=\theta -\dfrac{\alpha}{B}\displaystyle\sum_{i=1}^B\nabla_\theta\ell(h_\theta(x^{(i)}),y^{(i)})
$$

其中 batch 形式之前推导的结果如下：

$$
\dfrac{\partial}{\partial\theta}\ell_{ce}(X\theta,y)=X^T(Z-I_y)
$$

$$
Z=\text{normalize}(\exp(X\theta))
$$

 首先给定输入样本 $XX\in\mathbb{R}^{M\times n}$ ，其中 $M $ 是样本总数， $n$ 是特征维度，然后给定这些样本的标签 $y\in \mathbb{R}^{M\times 1}$ ，以及当前的初始参数 $\theta\in\mathbb{R}^{n\times k}$ ，首先我们通过 batch size （记为 $m$ ）以及总数计算出来要执行的 iteration（所有样本遍历一次），然后处理出来一个 $X\in\mathbb{R}^{m\times n}$ ，代码为 `X = XX[i*batch: (i+1)*batch]`，那么 $X^T\in\mathbb{R}^{n\times m}$ 即为 `X.T`，然后算 $Z\in\mathbb{R}^{m\times k}$ ，即 `xt = np.exp(np.matmul(X, theta))` 以及 `Z = xt / xt.sum(axis=1, keepdims=True)` ，按行 normalize 的时候注意 `sum` 那里的 `axis=1` 的参数， $I_y\in\mathbb{R}^{m\times k}$  实际上是从 $y\in\mathbb{R}^{m\times 1}$ 构造出来的 one-hot 向量（每个样本对应一个 one-hot 向量），该向量中只有标签对应的位置是 1，其余是 0，对应下面两行：

`Iy = np.zeros((batch, k), dtype=np.float32)`

`Iy[np.arange(batch), y[i*batch: (i+1)*batch]] = 1`

最后原地更新 $\theta$ 就行了：

`theta -= lr * np.matmul(X.T, (Z - Iy)) / batch` 


```python3
def softmax_regression_epoch(XX, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        XX (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    total = XX.shape[0]
    iteration = total // batch 
    k = theta.shape[1] # number of class
    for i in range(iteration):
        # \theta = \theta - \alpha * 1/m * X^T (Z - I_y)
        # \theta: n * k
        X = XX[i*batch: (i+1)*batch] # m * n
        xt = np.exp(np.matmul(X, theta)) # m * k
        Z = xt / xt.sum(axis=1, keepdims=True) # m * k
        Iy = np.zeros((batch, k), dtype=np.float32) # m * k
        Iy[np.arange(batch), y[i*batch: (i+1)*batch]] = 1
        theta -= lr * np.matmul(X.T, (Z - Iy)) / batch 
```

# 参考资料

[CMU 10-414/714: 机器学习系统 Deep Learning Systems](https://www.bilibili.com/video/BV1Rg4y137jH?spm_id_from=333.788.videopod.episodes&vd_source=aab0831003bc6e9f85163bf5cc0f8408&p=3)

