---
title: 如何拟合一个分布
keywords: "generative model"
---

## 引论

在上一篇文章中，我们的核心观点是采样是一种运算，它把简单随机分布中抽出的样本作为输入，输出值就作为采样结果。
我们定义的运算形式决定了采样服从什么分布。
同时我们也提出了一个会在本文中回收的伏笔：**将采样运算的部分或全部内容，用神经网络来替代，如何训练**。

以下行文中，若无特殊说明，神经网络参数组记为 $\theta$。
因为神经网络参与了采样所对应的运算的一部分，这也就意味着说最终采样的结果的分布，会随着网络参数 $\theta$ 的变化而变化，也就是说最终输出的分布是依赖于参数 $\theta$ 的选取的，所以我们不妨把此时模型所影响的分布的 pdf/pmf 记为 $p_\theta$。

我们最终想达到的效果是用 $p_\theta$ 来产生真实可用的数据，也即寻找足够好的 $\theta$ 使得 $p_\theta$ 接近于 $p_{\text{data}}$，后者 $p_{\text{data}}$ 表示的是我们获得真实数据的分布(为简化记号，若无特殊说明，后文中都用 $p$ 来表示获得真实数据的分布，省略掉下标 $\text{data}$)。
寻找 $\theta$ 的一类方式，就是构造一个以 $p_\theta, p$ 为输入的函数 $L$，它满足 $p_\theta, p$ 越接近，值越小，然后我们使用各种优化方法，以 $\theta$ 为优化变量，最小化 $L$ 即可满足用 $p_\theta$ 拟合 $p$ 的效果。

在本文中，我们就会来讨论 $L$ 的如何选取问题，展现出“拟合一个分布”和“拟合一个函数” 的差异点。

## 把 pdf/pmf 当成普通函数来拟合

记 $\mathcal{X}$ 为样本抽出的集合，$p, p_\theta$ 都首先是一个 $\mathcal{X} \to \mathbb{R}$ 的函数。
那从函数拟合的角度，我们不妨选

$$
\begin{aligned}
    L &= \int_{x \in \mathcal{X}} d(p_\theta(x), p(x)) \mathrm{d} x \\
    L &= \sum_{x \in \mathcal{X}} d(p_\theta(x), p(x))
\end{aligned}
$$

$d$ 是任意的 $\mathbb{R}$ 上的距离，比如欧氏距离、差的绝对值，更一般化地各种范数导出的距离...
两个式子选哪一个是依赖于我们处理的是离散样本还是连续样本，也就是说 $p$ 是 pdf 还是 pmf。

在实际训练中，我们一般不能得到所有可能的 $\mathcal{X}$ 的取值的样本，只能在有限的样本集上做所谓的经验估计(empirical estimation)，优化目标就变成：

$$
L \approx \frac{1}{|\mathcal{D}|} \sum_{x_n \in \mathcal{D}} d(p_\theta(x_n), p(x_n))
$$ 

但再细细想想，我们就会发现以上训练 loss 的问题。
按照上面的训练方式，我们需要准备的数据的形式是 $\left(x_n, p(x_n)\right)$ 这样的配对数据。
问题正出在这，我们很难发现说有数据集的提供形式是特征组 $x_n$ 和它的概率/概率密度值 $p(x_n)$。
通常我们只会得到一系列样本 $x_1, x_2, \dots, x_n$，然后我们知道它们都是从 $p$ 中采样出来的（更严格但是常见的描述是 $x_1, x_2, \dots, x_n$ 独立同分布从 $p$ 中采样）。

最直观的函数拟合的尝试的失败，启示着我们需要**更精巧的损失函数构造，来解决可以从 $p$ 中采样，但不能获知具体的 $p$ 的值的现状**。

## 极大似然估计

本节中，我们会逐渐推导一个能符合上面要求的损失函数，这个最小化这个损失函数也等价于在做类似极大似然的事，但我们会从分布拟合的角度重新认识它。

### 蒙特卡洛近似

我们先考虑一个更符合直觉的问题：估计 $p$ 的期望，当然也是在只能得到一系列抽样后样本 $x_1, x_2, \dots, x_n$，不能得到对应的 $p$ 的情景下。

对连续型随机变量和离散型随机变量，期望的定义是

$$
\begin{aligned}
    \mathbb{E}_{X \sim p(x)}\left[ X \right] = \int_{X \in \mathcal{X}} p(x) x \mathrm{d}x \\
    \mathbb{E}_{X \sim p(x)}\left[ X \right] = \sum_{X \in \mathcal{X}} p(x) x \mathrm{d}x
\end{aligned}
$$

可以看到，同样有 $p(x)$ 出现在积分/求和式中。下文中若无特殊说明，讨论的方法都是在连续型随机变量和离散型随机变量都适用的，为书写方便，涉及到需要展开定义式时，都默认用连续型随机变量和它积分形式的定义。

但我们对于期望的直觉理解就是“平均值”，所以即使没有 $p$ 的出现，只有一堆样本，我们也能很直觉的用如下定义的 $\bar{x}$ 作为分布期望的估计。

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i
$$

从大数定律，我们知道，当 $n \to \infty$ 时， $\bar{x}$ 收敛于 $\mathbb{E}\left[ X \right]$，确保了用 $\bar{x}$ 近似 $\mathbb{E}\left[ X \right]$ 的合理性。

结合期望的其他性质，可有推论：对任意函数 $f$ (忽略掉一些对函数的技术性要求)， $\mathbb{E}\left[ f(X) \right] = \int_{X \in \mathcal{X}} p(x) f(x) \mathrm{d}x$，同时它也可以用 $\bar{y} = \frac{1}{n} \sum_{i=1}^n y_i = \frac{1}{n} \sum_{i=1}^n f(x_i) $ 来近似。
我们从使用的角度把以上的话返过来说，就是：如果我们要求一个形如 $\int_{X \in \mathcal{X}} p(x) f(x) \mathrm{d}x$ 的式子，那我们可以仅使用多次独立采样得到的样本 $x_1, x_2, \dots x_n$，计算 $\frac{1}{n} \sum_{i=1}^n f(x_i)$ 来做近似。
这也就是数学家斯坦尼斯·乌拉姆在蒙特卡洛这一地点开发的[蒙特卡洛近似法](https://en.wikipedia.org/wiki/Monte_Carlo_method)。

$f$ 的任意性意味着此方法的适用面之广，灵活度之高。
可以尝试着将其用于解决我们在上一节中构造损失函数失败时所面对的问题。
结合蒙特卡洛方法所要求的式子形式，我们会发现一种解决方案是，把损失函数构造出如下形式：

(eq-mc_loss_int)=
$$
L = \mathbb{E}_{x \sim p(x)}[ f(x, p_\theta(x)) ] = \int_{X \in \mathcal{X}} p(x) f(x, p_\theta(x)) \mathrm{d}x
$$

也即 $L$ 中 $p(x)$ 是能被完全分离出来作为待积分式子用乘法作用的加权因子。
此时 $L$ 无需 $p(x)$ 的具体值只用从 $p$ 中采样出来的样本值，也能做近似

$$
L \approx \frac{1}{n} \sum_{i = 1}^n f(x_i, p_\theta(x_i))
$$

如果采用这种形式的损失函数，那我们就能用有限多个样本点值，来做损失函数的计算，进而利用损失函数优化网络参数了。
那接下来的问题是：怎么具体构造一个有上式样式的损失函数。

### 过渡小节

到这里，我们对 $L$ 就有了两点要求：
1. $L$ 应当在 $p_\theta$ 与 $p$ 越接近时越小。
2. $L$ 应有如式 @eq-mc_loss_int 的形式。

构造的主要难点是2，因为式 @eq-mc_loss_int 中 $p$ 和 $p_\theta$ 的地位不再是平等的了，很多符合我们直觉的对称二元函数（比如各种距离）都大概是用不了了。
同时，从另一个侧面也可以看到为了构造符合条件的 $f$, 我们不再能将$p_\theta$ 视为普通函数（即无视 $p_\theta$ 应该逐点大于等于零，积分为1的条件），反向说明如下：

假设我们将 $p_\theta$ 视为普通函数，也就是 $p_\theta$ 在 $\mathcal{X}$ 上各点的取值互不约束，考虑最小化式 @eq-mc_loss_int 的一种思考方式是，被积分项是两项相乘，因为第一项 $p(x) \geq 0$，所以只需要在固定点 $x$上 都最小化 $f(x, p_\theta(x))$，那么二者相乘，积分起来一定是最小的。
逐点最小化 $f(x, p_\theta(x))$ 这个过程得到的 $p_\theta$ 只会与 $x$ 有关，与 $p(x)$是无关的，所以优化的结果就不可能是在每个 $x$ 上 $p(x) = p_\theta(x)$，也即无法达到我们想要的分布拟合的效果。

反过来，当考虑 $p_\theta$ 需要是一个 pdf/pmf，也即考虑 $p_\theta$ 在各点大于等于零，求和/积分为 $1$ 的条件的时，逐点优化 $f(x, p_\theta(x))$ 的思路解法就不可取了，推理过程如下：
假设在固定点 $x$ 上， $f(x, p_\theta(x))$ 减小需要 $p_\theta(x)$ 的减小，所以只考虑 $x$ 点的状态，你会减小 $p_\theta(x)$ 的值，因为但因为约束的存在： $p_\theta(x)$ 不能无限制地减小，最多小到零；同时，因为求和/积分为 $1$ 这个条件的存在，减小当前点的 $p_\theta(x)$ 势必会增大其他点的 $p_\theta(x)$，别的点（记为 $x'$） 上的 $p_\theta(x')$ 的增大，会导致 $f(x', p_\theta(x'))$ 的变化，进而导致 $L$ 的变化是在不同点 $x, x', \dots$ 的 $f$ 的值的变化的求和/积分。
这样的约束下，才**有可能**保证选择合适的 $f$ 在这样“此消彼长”的竞争下，$L$ 的最小值点会是 $p_\theta = p$。

以上的描述可能还有些抽象，我们举一个具体的示例（但示例最终不满足最优解是 $p_\theta = p$）看看，考虑一个简单的 $f$， $f(x, p_\theta(x)) = p_\theta(x)$，则 $L$ 就可以具体写为:

$$
L = \int_{x \in \mathcal{X}} p(x) p_\theta(x) \mathrm{d}x 
$$

如果只把 $p_\theta(x)$ 当成普通的函数，那么显然，在每个点 $x$，$p_\theta$ 越小，最后的积分值 $L$ 也就越小，我们就会希望 $p_\theta(x)$ 一路减小，小到零，小到负数，小到趋近负无穷，从而我们会看到 $L$ 在这样的考虑下其实是无下界的。
这样，我们也就明显感知到不考虑 $p_\theta$ 是 pdf/pmf 这个条件的不合理性。
如果考虑约束 $p_\theta(x) \geq 0, \forall x \in \mathcal{X},  \int_{x \in \mathcal{X}} p_\theta(x) \mathrm{d}x = 1$，最小化 $L$ 的求解思路就会发生变化。

一种启发式地解法是：注意到 $p$ 是给定的函数，$p_\theta$ 逐点大于等于0，求和/积分为 $1$，从而能将 $L$ 看为对 $p(x)$ 的加权平均，不同的 $p_\theta(x)$ 的取法，就代表着分配了不同的权重组。
那凭借我们对加权平均的理解，我们可以将 $p(x)$ 看成是 $p(x) = p_{\min} + \Delta(x)$，即任意点的 $p(x)$ 可以看成是最小点$p_{\min}$ 再加上一个大于等于零的增量 $\Delta$ (这里假定 $p(x)$ 有最小值，但其实一般情况下不能保证能取到)。
从而看出只有当我们给最小的那个值分配满的权重 $1$，其他部分分配权重 $0$，才能将加权平均值最小化。
也即对 $L$ 做变换和放缩：

$$
\begin{aligned}
   L &= \int_{x \in \mathcal{X}} p(x) p_\theta(x) \mathrm{d}x \\
     &= \int_{x \in \mathcal{X}} (p_{\min} + \Delta(x)) p_\theta(x) \mathrm{d}x \\
     &= \int_{x \in \mathcal{X}} p_{\min} p_\theta(x) \mathrm{d}x +\int_{x \in \mathcal{X}} \Delta(x) p_\theta(x) \mathrm{d}x \\
     &= p_{\min}  \int_{x \in \mathcal{X}} p_\theta(x) \mathrm{d}x + \int_{x \in \mathcal{X}} \Delta(x) p_\theta(x) \mathrm{d}x \\
     &= p_{\min} + \int_{x \in \mathcal{X}} \Delta(x) p_\theta(x) \mathrm{d}x \\
     &\geq p_{\min} + 0 \\
     &= p_{\min}
\end{aligned}
$$

最后一步的放缩是根据 $\Delta(x) \geq 0$ 当且仅当 $p(x) = p_{\min}$ 时取等，以及 $p_\theta \geq 0$ 两个条件放缩得到的。
并且也可以看到，我们的放缩的取等的条件是 $p_\theta$ 在所有非 $p(x) = p_{\min}$ 的点 $x$ 上都取 $0$，即不在这些点分配权重/概率/概率密度值（这里有数学描述上不严谨的地方）。
所以最终，取 $f(x, p_\theta(x)) = p_\theta(x)$时，最小 $L$ 所对应的 $p_\theta$ 是集中在 $p_{\min}$ 所对应的那些点 $x$ 的分布。
对比不考虑 $p_\theta$ 是 pdf/pmf 这个条件时 $L$ 无下界的情形，可以看到这个约束发挥的作用。

但也可以看到，我们选取的这个简单的 $f$，所导出的最优解，并不是我们想要的 $p_\theta = p$，还需要另寻他路，构造别的 $f$。

### 交叉熵

这里我们机械降神般地给出符合之前要求的 $f$ 的构造。
我个人认为从两个条件正向想出这个构造最直接的办法是变分法，但变分超过了本文承诺的前置知识范围，所以只在最后简单说明一下。
一个满足条件的 $f$ 是

$$
f(x, p_\theta(x)) = -\log p_\theta(x)
$$

我们来验证 $L$ 的最小值解就是 $p_\theta = p$。
写出此时的目标函数 $L$ 的具体形式：

(eq-cross_entropy)=
$$
L = \int_{x \in \mathcal{X}} p(x) \left[-\log p_\theta(x) \right] \mathrm{d}x
$$

下面我们将按 二值离散分布，消元法 -> 离散分布，消元法 -> 离散分布，拉格朗日乘子法 -> 离散分布，Jense 不等式法 -> 一般分布， Jense 不等式法 -> 一般分布，变分法，来介绍求取式 @eq-cross_entropy 最小值的办法，验证“$L$ 的最小值解就是 $p_\theta = p$” 的陈述。
这样排版的目的是希望能给不同基础的读者都有层层递进之感，不同基础的读者可以按需跳转到对应的位置。

直接考虑一般性的问题似乎无从下手，我们先从稍微具体而简单的分布考虑：

**二值离散分布，消元法**

不妨设随机变量可取的二值是 ${0, 1}$， 此时目标函数可具体写出，并做一定的化简

$$
\begin{aligned}
    L &= \sum_{x = 0, 1} p(x) \left[ -\log p_\theta(x) \right] \\
      &= p_0 \left[ -\log p_{\theta, 0} \right] + (1 - p_0) \left[ -\log (1 - p_{\theta, 0}) \right]  
\end{aligned}
$$

其中 $p_0$ 是 $p(x = 0)$ 的简写， $p_{\theta, 0}$ 是 $p_{\theta}(x = 0)$ 的简写，从而我们可以看到，原来的“通过调整 $p_\theta$ 最小化 $L$”的问题，可以依照上式末尾，取优化变量为 $p_{\theta, 0}$，并直接求导就能得到极值点情况，也就是

$$
\begin{aligned}
    & \frac{\partial L}{\partial p_{\theta, 0}} = -\frac{p_0}{p_{\theta, 0}} + \frac{1 - p_0}{1 - p_{\theta, 0}} = 0 \\
    \Rightarrow & \frac{p_0}{p_{\theta, 0}} = \frac{1 - p_0}{1 - p_{\theta, 0}} \\
    \Rightarrow & p_{\theta, 0} = p_0
\end{aligned}
$$

从而 $p_{\theta, 1} = 1 - p_{\theta, 0} = 1 - p_0 = p_1$，那么两个二值分布列 $p, p_\theta$ 就相等了。
在这个解法中，约束 $p_0 + p_1 = 1, p_{\theta, 0} + p_{\theta, 1} = 1$ 是移项消元，代入目标函数中，来间接地被纳入进来。

这样解出来的 $p_{\theta, 0}$ 只能保证它是极值点，还需要说明它是极小值点，且这个极小值点是最小值点。
这通过求二阶导数判断导函数单调性/画图法/不等式瞪眼放缩法，都可以得到，就留给读者简单验证了。


**一般离散分布，消元法**

这时，不妨假设随机变量可取的值是 $1, 2, \dots, n$，并类似地简记 $p(x = i)$ 为 $p_i$, $p_\theta(x = i)$ 为 $p_{\theta, i}$，此时目标函数可具体写为

$$
L = \sum_{x = 1}^n p(x) \left[ -\log p_\theta(x) \right] = -\sum_{i = 1}^n p_i \log p_{\theta, i}
$$

我们有的关于优化变量 $p_{\theta, i}$ 的约束是 $ p_{\theta, i} \geq 0$ 和 $\sum_{i=1}^n p_{\theta, i} = 1$。
因为求和式中我们对 $p_{\theta, i}$ 取了对数，考虑对数的定义域，则只要我们求出了能回带到原式的 $p_{\theta, i}$ 其实就满足了更强的约束 $p_{\theta, i} > 0$，自然也就满足了 $p_{\theta, i} \geq 0$。

所以我们的重点就是处理等式约束 $\sum_{i=1}^n p_{\theta, i} = 1$。
那按照同样的逻辑，有一个等式约束，就用它来代入约简掉一个自由变量，这里不妨取要代入约简的变量是 $p_{\theta, n}$，则将约束处理如下

$$
\sum_{i=1}^n p_{\theta, i} = 1 \Rightarrow p_{\theta, n} = 1 - \sum_{i=1}^{n - 1} p_{\theta, i}
$$

代入目标函数则有

$$
L = -\sum_{i = 1}^{n - 1} p_i \log p_{\theta, i} - p_n \log\left[ 1 - \sum_{i=1}^{n - 1} p_{\theta, i} \right]
$$

上式的优化变量为 $p_{\theta, i}, \, (i = 1,2,\dots , n - 1)$。
则为求极值点，则需要考虑 $n - 1$ 个变量关于 $L$ 的偏导都为 $0$ 即

$$
\begin{aligned}
                & \frac{\partial L}{\partial p_{\theta, i}} = 0 \quad i = 1,2,\dots, n - 1 \\
    \Rightarrow & -\frac{p_i}{p_{\theta, i}} + \frac{p_n}{1 - \sum_{i=1}^{n - 1} p_{\theta, i}} = 0 \quad i = 1,2,\dots, n - 1 \\
    \Rightarrow & \frac{p_1}{p_{\theta, 1}} = \frac{p_2}{p_{\theta, 2}} = \cdots = \frac{p_{n - 1}}{p_{\theta, n - 1}} = \frac{p_n}{1 - \sum_{i=1}^{n - 1} p_{\theta, i}} \\
    \Rightarrow & \frac{p_1}{p_{\theta, 1}} = \frac{p_2}{p_{\theta, 2}} = \cdots = \frac{p_{n - 1}}{p_{\theta, n - 1}} = \frac{p_n}{p_{\theta, n}}
\end{aligned}
$$

重要的自然是前面的 $n$ 个比例相等，$\frac{p_1}{p_{\theta, 1}} = \frac{p_2}{p_{\theta, 2}} = \cdots = \frac{p_{n - 1}}{p_{\theta, n - 1}}$ 可以转为 $p_{\theta, 1} = k p_1, p_{\theta, 2} = k p_2, \dots p_{\theta, n - 1} = k p_{n - 1}, p_{\theta, n} = k p_{n}$。
综合考虑 $\sum_{i = 1}^n p_i = 1$ 以及 $\sum_{i = 1}^n p_{i, \theta} = 1$ 这两个已知条件，不难解出 $k = 1$，从而我们知道极值点是 $p_\theta = p$。

同样我们还需要说明：1. 这个极值点是极小值点，2. 极小值点就是最小值点，但也留个读者验证。

从这两个比较繁复的消元求极值的办法的细节推导，我们可以看到正是 $\sum_{i=1}^n p_{i, \theta} = 1$ 的约束，使得说每个 $p_{i, \theta}$ 不能无约束地增大或减小，某一个 $p_{i, \theta}$ 增大，就会导致其他 $p_{i, \theta}$ 减小，从而导致 $L$ 的非单调变化。

但是这样消元的方法有点过于考验对约束条件的处理手法了，我们用更通用的方法来尝试一下

**一般离散分布，拉格朗日乘子法**

重写此时的优化问题为

$$
\begin{aligned}
    \min_{p_{\theta, i}} \, & L = \sum_{i = 1}^n p_i \left[ -\log p_{\theta, i} \right] \\
    \text{s.t.} & \sum_{i=1}^n p_{\theta, i} = 1
\end{aligned}
$$

很清晰地看到，我们拥有 $n$ 个优化变量 $p_{\theta, i}, \, (i = 1,2,\dots, n)$，一个约束条件： $\sum_{i=1}^n p_{\theta, i} = 1$。

那么对这个等式约束引入拉格朗日乘子 $\mu$, 先求优化问题

$$
\min_{p_{\theta, i}, \mu} \, & \mathcal{L} = \sum_{i = 1}^n p_i \left[ -\log p_{\theta, i} \right] - \mu \left[ \sum_{i=1}^n p_{\theta, i} - 1\right]
$$

$$
\begin{aligned}
    & \frac{\partial \mathcal{L}}{\partial p_{\theta, i}} = -\frac{p_i}{p_{\theta, i}} + \mu = 0 \quad i = 1,2,\dots, n \\
    \Rightarrow & \frac{p_1}{p_{\theta, 1}} = \frac{p_2}{p_{\theta, 2}} = \cdots = \frac{p_{n - 1}}{p_{\theta, n - 1}} = \frac{p_n}{p_{\theta, n}} = \mu \\
\end{aligned}
$$

即对不同的随机变量可取值$i$ ，目标分布的 $p_i$ 和模型分布的 $p_{\theta,i}$ 比例不变，也即 $p_{\theta,i} = \frac{1}{\mu} p_i$，则也同样地，从 $\sum_{i=1}^n p_{\theta, i} = 1$, $\sum_{i=1}^n p_{i} = 1$ 出发，有 $p_i = p_{\theta,i}$

从而验证了 $p_\theta = p$ 是原优化问题 @eq-cross_entropy 的最小值点。

**一般离散分布，Jensen 不等式法**

再换用 Jensen 不等式来求解一般离散分布交叉熵的最小值点，是为了给一般连续分布的情形做铺垫。

首先通过求两次导数，我们很容易验证 $f(x) = -\log(x) $ 在它的定义域上是（严格）凸函数。
则用推广的 Jense 不等式，对于任意的权重 $w_1, w_2, \dots, w_n \geq 0$，且满足 $\sum_{i = 1}^n w_i = 1$

$$
\sum_{i = 1}^n w_i f(x_i) \geq f\left( \sum_{i = 1}^n w_i x_i \right)
$$

当且仅当 $x_1 = x_2 = \dots = x_n$ 等号成立。

那么看到在交叉熵里，前面的 $p_i$ 就是想我们所想的“权重” $w_i$ 那样，满足大于等于0，求和为1的条件。
后面的项 $-\log p_{\theta, i}$ 也和已经被验证为凸函数的 $f(x) = -\log(x) $ 有些相似。
这里可能很容易直接就拍脑袋 “可根据 Jensen 不等式放缩”

$$
\sum_{i = 1}^n p_i \left[ -\log p_{\theta, i} \right] \geq -\log \left[ \sum_{i = 1}^n p_i p_{\theta, i} \right]
$$

但这样并不能做下去，而且放缩的逻辑其实也是有误的。
具体来说有两个问题：

1. 放缩后的式子还是显含 $p_\theta$，即放缩后还是一个关于 $p_\theta$ 可变的函数**而不是常数值**。不能说明任何原式的最值点情况。
2. 证明了是凸函数的是 $f(x) = -\log x$，可不是 $f(x) = -\log p_\theta(x)$ ($x$ 可取 $1, 2, \dots n$)。我们这样的放缩真有可能不成立。


接下来我们来同时解决这两个问题。

先考虑第一个问题。
注意到优化变量仅仅是 $p_{\theta}$，加减一些纯含 $p$ 的项并不会改变最优值取到时 $p_{\theta}$ 的情况。

$$
\min_{p_{\theta}} \sum_{i = 1}^n p_i \left[ -\log p_{\theta, i} \right] \Leftrightarrow \min_{p_{\theta}} \sum_{i = 1}^n p_i \left[ -\log p_{\theta, i} \right] - \sum_{i = 1}^n p_i \left[ -\log p_i \right]
$$

新的目标函数，通过加减合并项和对数的性质，可以再转换为
$$
\min_{p_{\theta}} \sum_{i = 1}^n p_i \left[ -\log \frac{p_{\theta, i}}{p_i} \right]
$$

那这时，一个可能的放缩办法就是

$$
\begin{aligned}
    \sum_{i = 1}^n p_i \left[ -\log \frac{p_{\theta, i}}{p_i} \right] &\geq -\log \left[ \sum_{i = 1}^n p_i \frac{p_{\theta, i}}{p_i} \right] \\
    & = -\log \left[ \sum_{i = 1}^n p_{\theta, i} \right] \\
    & = -\log \left[ 1 \right] \\
    & = 0
\end{aligned}
$$

这样就解决了之前提出的第一个问题了，但第二个问题看上去似乎更严重了，我们现在的放缩看上去是在 $f(x) = -\log \frac{p_\theta(x)}{p(x)}$ 上做的，这个东西还更不好说它是不是凸函数了。

所以我们马上就来考虑第二个问题，我们换个角度，在随机变量的角度来看待。

原始的随机变量是 $X$，它可取的随机值范围为 $1, 2, \dots, n$。
现在我们考虑一个从 $X$ 派生的随机变量 $Y$，它的派生方法是 $Y = \text{some function}(X)$。
只不过这个函数比较特殊，我们选的函数就是概率质量函数 $\frac{p_\theta (X)}{p(x)}$，它确实也是一个函数，输入一个 $1, 2, \dots, n$ 上的值，返回随机变量 $X$ 取输入值时，在两个分布看待下的概率值之比，而已。
也即 $Y = \frac{p_\theta (X)}{p(X)}$。

那么利用 $f(y) = -\log y$是凸函数对应的 Jensen 不等式，它确实能保证 $\mathbb{E}_{Y} \left[ -\log Y \right] \geq -\log \mathbb{E}_Y \left[ Y \right]$。

但从 $Y, X$ 的派生关系，我们也可以有

$$
\begin{aligned}
    \mathbb{E}_Y \left[ -\log Y \right] &=& \mathbb{E}_X \left[ -\log \frac{p_\theta (X)}{p(X)} \right] &=& \sum_{i = 1}^n p_i \left[ -\log \frac{p_{\theta, i}}{p_i} \right] \\
    \mathbb{E}_Y \left[ Y \right] &=& \mathbb{E}_X \left[ \frac{p_\theta (X)}{p(X)} \right] &=& \sum_{i = 1}^n p_i \frac{p_{\theta, i}}{p_i}
\end{aligned}
$$

所以，上面的放缩是完全成立的。
从而我们知道，(最左式也即$p, p_\theta$ 之间的 KL 散度)

$$
\sum_{i = 1}^n p_i \left[ -\log \frac{p_{\theta, i}}{p_i} \right] \geq 0 \Rightarrow \sum_{i = 1}^n p_i \left[ -\log p_{\theta, i} \right] \geq \sum_{i = 1}^n p_i \left[ -\log p_i \right]
$$

当且仅当 $\frac{p_{\theta, 1}}{p_1} = \frac{p_{\theta, 2}}{p_2} = \dots = \frac{p_{\theta, n}}{p_n}$ 时取等，取等条件不难化简后得到它等价于 $p_\theta = p$。
也即说明了 @eq-cross_entropy 最小值点就是 $p_\theta = p$。

**一般连续分布，Jensen 不等式法**

有了上面的引入后，就很容易过度到一般连续分布的情形了。
通过取极限（其实这里也有伪证的成分，需要加一些额外条件），我们知道对于一个连续型随机变量 $Y$ 和一个凸函数 $f$，仍然有

$$
\mathbb{E}_Y \left[ f(Y) \right] \geq f\left( \mathbb{E}_Y \left[ Y \right] \right)
$$

那类似地复刻上面的流程，我们不难得到如下放缩推导过程。

取随机变量 $Y = \frac{p_\theta (X)}{p(X)}$，则 @eq-cross_entropy 可以做如下变换放缩。

$$
\begin{aligned}
    \int_{x \in \mathcal{X}} p(x) \left[ -\log p_\theta(x) \right] \mathrm{d}x &= \mathbb{E}_{X} \left[ -\log p_\theta(X) \right] \\
    &= \mathbb{E}_X \left[ -\log p_\theta(X) - (-\log p(X)) + (-\log p(X)) \right] \\
    &= \mathbb{E}_X \left[ -\log \frac{p_\theta(X)}{p(X)} \right] + \mathbb{E}_X \left[-\log p(X) \right] \\
    &= \mathbb{E}_Y \left[ -\log Y \right] + \mathbb{E}_X \left[ -\log p(X) \right] \\
    &\geq -\log  \mathbb{E}_Y \left[ Y \right] + \mathbb{E}_X \left[ -\log p(X) \right] \\
    &=  -\log  \mathbb{E}_X \left[ \frac{p_\theta (X)}{p(X)} \right] + \mathbb{E}_X \left[ -\log p(X) \right] \\
    &= -\log \int_{x \in \mathcal{X}} p(x) \frac{p_\theta(x)}{p(x)} \mathrm{d}x + \mathbb{E}_X \left[ -\log p(X) \right] \\
    &= -\log \int_{x \in \mathcal{X}} p_\theta(x) \mathrm{d}x + \mathbb{E}_X \left[ -\log p(X) \right] \\
    &= -\log 1 + \mathbb{E}_X \left[ -\log p(X) \right] \\
    &= \mathbb{E}_X \left[ -\log p(X) \right]
\end{aligned}
$$

当且仅当 $\frac{p_\theta(x)}{p(x)}$ 在 $x \in \mathcal{X}$ 上均为常数时取等，结合 $p, p_\theta$ 两个概率密度积分为 $1$ 的条件出发，不难得到取等条件等价于 $p_\theta = p$。

从而我们知道 @eq-cross_entropy 最小值点就是 $p_\theta = p$，最小值具体为 $\mathbb{E}_X \left[ -\log p(X) \right]$

**一般连续分布，变分法**

这个方法是拉格朗日乘子法的推广，也是在一般连续分布的情形下的一般性求解方法。
只是考虑到变分法不在本文承诺的前置知识范畴中，所以放在最后。

优化问题是

$$
\begin{aligned}
    \min_{p_{\theta}} \, & L = \int_{x \in \mathcal{X}} p(x) \left[ -\log p_\theta(x) \right] \mathrm{d}x \\
    \text{s.t.} & \int_{x \in \mathcal{X}} p_\theta(x) \mathrm{d}x = 1
\end{aligned}
$$

通过引入拉格朗日乘子 $\mu$，我们可以得到拉格朗日函数$\mathcal{L}$，并将问题变为无约束优化问题

$$
\min_{\theta, \mu} \, & \mathcal{L} = \int_{x \in \mathcal{X}} p(x) \left[ -\log p_\theta(x) \right] \mathrm{d}x - \mu \left[\int_{x \in \mathcal{X}} p_\theta(x) \mathrm{d}x - 1 \right] \
$$

求 $\mathcal{L}$ 关于函数 $p_\theta$ 的变分，即为

$$
\begin{aligned}
    \delta L &= \int_{x \in \mathcal{X}} p(x) \left[ -\frac{1}{p_\theta(x)} \delta p_\theta(x) \right] \mathrm{d}x - \mu \left[\int_{x \in \mathcal{X}} \delta p_\theta (x) \mathrm{d}x \right] \\
    &= \int_{x \in \mathcal{X}} \left[ -\frac{p(x)}{p_\theta(x)} - \mu  \right] \delta p_\theta(x) \mathrm{d}x
\end{aligned}
$$

在极值点处，对于任意的 $\delta p_\theta(x)$，都有 $\delta L = 0$，从而可知在极值点处必然有
$$
-\frac{p(x)}{p_\theta(x)} - \mu = 0
$$

从而类似的联合概率密度积分为一的约束条件，我们可以得到极值点上的 $p_\theta$ 满足 $p_\theta = p$

### 交叉熵、KL 散度、极大似然估计的关系

至此，我们已经知道了 $p, p_\theta$ 之间的交叉熵 @eq-cross_entropy，在取优化变量为 $p_\theta$ 时，最小值点就是 $p_\theta = p$。

并且从上面的各种推导过程中摘取中间结论，或者现在独立推导，我们不难看到这样的等价关系

$$
\begin{aligned}
    & \min_{p_\theta}  \int_{x \in \mathcal{X}} p(x) \left[ -\log p_\theta(x) \right] \mathrm{d}x \\
    \Leftrightarrow & \min_{p_\theta} \int_{x \in \mathcal{X}} p(x) \left[ -\log \frac{p_\theta(x)}{p(x)} \right] \mathrm{d}x \\
\end{aligned}
$$

而后者正是 $p, p_\theta$ 之间的 KL 散度，所以说在优化变量是 $p_\theta$ 的情况下，最小化交叉熵和最小化 KL 散度是等价的。

那么再想想我们推导出应该用交叉熵来做损失函数的理由，是要找一个可以用蒙特卡洛近似的损失函数。
那么现在我们就来给交叉熵做一下蒙特卡洛近似。

$$
\begin{aligned}
L &= \int_{x \in \mathcal{X}} p(x) \left[ -\log p_\theta(x) \right] \mathrm{d}x \\
  &= \mathbb{E}_{X \sim p(x)} \left[ -\log p_\theta(X) \right] \\
  &\approx \frac{1}{n} \sum_{x_i \sim p(x)} \left[ -\log p_\theta(x_i) \right] \\
  &= -\frac{1}{n} \log \left[ \prod_{x_i \sim p(x)} p_\theta(x_i)\right]
\end{aligned}
$$

不难看出，做蒙特卡洛近似后的式子，就是所有因为蒙特卡洛近似而采出来的样本组，在 $p_\theta$ 上的对负对数似然，（乘了一个系数）。
也就是在说最小化交叉熵，在蒙特卡洛近似后就等价于最小化负对数似然，也就等价于最大化似然。

## 总结

在本文中，我们介绍了在分布拟合场景下损失函数的构造方法，通过损失函数要方便做蒙特卡洛近似，最小值点应该是 $p_\theta = p$， 引出了交叉熵，并分析了交叉熵的性质，最后说明了最小化交叉熵和最小化 KL 散度、极大似然估计的等价性。

同时也需要点出的是，最小化交叉熵仅仅是一种可行的做分布拟合的损失函数，但不是唯一的。
这里着重介绍交叉熵，一是因为它相对简单，推出函数后，做便于蒙特卡洛和最小值点条件的验证都很方便，利于我们强调文章的中心主旨；二是因为它确实能串联起目前大部分主流的生成模型的训练过程。

本篇文章读完后，希望读者对最抽象笼统的“生成模型该怎么训练”有一个认知，从下一篇文章开始，我们就会深入到各个不同的主流生成模型训练方法（如 AR、VAE、GAN、Diffusion 等等），详细介绍这些模型的细节，并回收上一篇文章中的其他伏笔。

## 彩蛋

### 判别模型生成模型的分野

在一些观念中，生成式模型和判别式模型是需要做概念上的区分的。
并且在本篇文章中，我们给出的第一个失败尝试，就是用 $p(x), p_\theta(x)$ 的距离来做损失，这种看上去承袭自线性回归或者说所谓判别式模型的尝试。
但在本节中，我想核心传达的观点是 **判别模型和生成模型没有分野**，全看你如何考虑问题。

重新抄写一遍，用交叉熵训练生成模型的流程：

1. 我们有一个神经网络 $g_\theta$，它的所有的或显式或隐式的输入里有一些是随机变量 $z$，这些随机变量是从一个分布 $p(z)$ 中采样出来的，从而网络的输出也会有随机性。
2. 因为网络的输出有随机性，而且依赖于网络参数 $\theta$ 的选取，所以可以认为现在我们有一个随模型参数 $\theta$ 变化的分布 $p_\theta(x)$。
3. 为了做一个好的生成模型，我们自然希望 $p_\theta(x)$ 能拟合 $p(x)$。
4. 因为我们不能直接得到 $p(x)$ 的值，但能从 $p(x)$ 中抽样，所以我们选择一个好蒙特卡洛近似的损失函数 —— 交叉熵，用它训练网络，即为解上式的优化问题，在实践中求解近似后的下式优化问题：

$$
\begin{aligned}
    \min_{\theta} \, \mathbb{E}_{X \sim p(x)} \left[ -\log p_\theta(X) \right] \\
    \min_{\theta} \, \frac{1}{n} \sum_{x_i \sim p(x)} \left[ -\log p_\theta(x_i) \right] \\
\end{aligned}
$$

另一方面，以（多元）线性回归为例，看上去它处理的问题完全不一样。
它是在说，我有一对变量 $(x, y)$和一堆数据 $(x_i, y_i)$，我希望通过这些数据，找到一个模型 $f_\theta$，使得 $f_\theta(x_i)$ 能接近 $y_i$。
也即用模型 $f_{\theta}$ 来拟合数据 $(x,y)$的潜在关系函数 $f$。
整个训练过程是在解优化问题：

$$
\min_{\theta} \, \frac{1}{n} \sum_{i = 1}^n \left( y_i - f_\theta(x_i) \right)^2
$$

看上去两个问题的出发点完全不同，优化问题的形式也完全不同。
但实际上我们可以把后者转换为前者。
为了配凑上形式，我们考虑给 $\left( y_i - f_\theta(x_i) \right)^2$ 添加一些完全无关于 $\theta$ 的项，使得优化问题能逐渐配凑成 $-\log p_\theta(x_i)$ 的样子，等价回交叉熵的样式，变换过程如下：

$$
\begin{aligned}
    \left( y_i - f_\theta(x_i) \right)^2 &= - \left[ -\left( y_i - f_\theta(x_i) \right)^2 \right] \\
    &= - \left\{ \log  \left[ e ^{-\left( y_i - f_\theta(x_i) \right)^2} \right] \right\} \\
\end{aligned}
$$

也就是说，如果 $p_\theta$ 形如 $ e^{-\left( y_i - f_\theta(x_i) \right)^2}$ 的样式（这里形如的意思是指 $p_\theta$ 可以是右式再乘上某个系数，用来保证 $p_\theta$ 积分起来是 $1$，满足分布的归一化条件），那此时用交叉熵来训练模型，就等价于原理的最小均方误差来训练模型。

再看看 $p_\theta$ 的样子，其实就是在说它是一个**关于 $y$ 的随机分布**，或者说是条件于给定的一个 $x$ 后 $y$ 的分布 $p_\theta(y | x)$，是均值为 $f_\theta(x)$，方差为 $\frac{1}{\sqrt 2}$ 的正态分布。
即我们认为 $y \sim \mathcal{N}(f_\theta(x), \frac{1}{\sqrt 2})$，然后在此基础上，采样了一堆 $(x_i, y_i)$ 来做分布拟合。

另外，这里也可以注意到，此时神经网络直接输出的量 $f_\theta(x)$ 不是 $p_\theta(x)$， 只是某个中间量 —— 高斯分布的均值，如果要把它当成生成模型来看，要想输出随机值，我们就需要根据网络直接输出的均值 $f_\theta(x)$ 和给定的方差 $\frac{1}{\sqrt 2}$ 来做随机采样。当然在判别为中心的场景下，我们一般不这么考虑而已，直接把 $f_\theta(x)$ 就当成最后的输出结果用。

除了观念上的统一之外，统一用生成模型的视角看待这些方法还有一个好处。
我们来看待一个经典八股题：“在 0-1 二分类问题中，不用 BCE 而强行用 MSE 给模型训练有什么不对的？”

这题八股标准回答类似于[这样](https://zhuanlan.zhihu.com/p/655331840)，侧重于通过梯度来分析。

但我一直觉得这样的回答有些过于唯象了，梯度分析只能给出什么“训得快，训得慢，不好收敛”的定性评价。
我们用生成模型的视角来看待这个问题。

用 BCE 训练的时候损失函数是：
$$
\begin{aligned}
    L &= \frac{1}{n} \sum_{i = 1}^n -\left[ y_i \log f_\theta(x_i) + (1 - y_i) \log (1 - f_\theta(x_i)) \right] \\
    &= \frac{1}{n} \left\{ \sum_{y_i = 1} -\log f_\theta(x_i) + \sum_{y_i = 0} -\log \left[1 - f_\theta(x_i) \right]\right\}
\end{aligned}
$$

用生成的角度来看，也就是模型在认为 $y$ 在给定 $x$ 的条件的条件分布是一个 0-1 二值分布，它取 1 的概率是 $f_\theta(x)$，取 0 的概率是 $1 - f_\theta(x)$。
那实际中 $y$ 也确实就是 $0, 1$ 两种取值，即确实也有 $p(y|x)$ 是一个二值分布。
在 BCE 损失下，我们假设的模型分布 $p_\theta$ 和真实分布 $p$ 是同一族函数，所以只要训练数据足够多，能无限缩小蒙特卡洛的误差，训练足够久，保证收敛性，确实能保证 $p_\theta$ 能完全拟合到 $p$ 上。

当换成 MSE 后，我们假设的模型分布就变成了 $p_\theta$ 是一个关于 $y$ 的条件分布，在给定 $x$ 的条件下均值为 $f_\theta(x)$，方差为 $\frac{1}{\sqrt 2}$ 的正态分布了。
那因为二者是不同类型的分布函数，自然最小值点就不可能使得 $p_\theta = p$。

但最小值点还是客观存在的。
我们就来硬求一下最小值点。
为了排除蒙特卡洛近似的影响和网络表达能力不够的影响，我们把优化变量视为 $f$，假设 ground truth 分布 $p$ 取 $1$ 的概率为 $p_1$，在交叉熵的解析式上求优化问题：

$$
\begin{aligned}
    & \min_{f} \, \mathbb{E}_{Y \sim p(y|x)} \left[ -\log p_\theta(Y|x) \right] \\
    \Leftrightarrow & \min_{f} \, L = p_1 (f - 1)^2 + (1 - p_1) (f - 0)^2 \\
\end{aligned}
$$

$\frac{\partial L}{\partial f} = 2(f - p_1)$，通过令 $\frac{\partial L}{\partial f} = 0$，解方程可以得到 $f = p_1$。
也就是说，虽然两个分布 $p_\theta$ 和 $p$ 不能完全贴合，但 $p_\theta$ 的均值，也即网络直接的输出 $f_\theta(x)$ 其实还是达成了拟合 $p_1$ 的效果。所以如果训网络的时候是 MSE 训，但用网络的时候还是把网络的直接输出诠释为取值为 $1$ 的概率，用 MSE 训练和用 BCE 损失训练的最优效果是一样的。

那再看训练过程的差异。
使用 BCE 的时候，损失函数对网络直接的输出，也即 sigmoid 层后的输出 $p_{\theta, 1}$ 和 sigmoid 层前的输出 $h$ 的梯度分别为：

$$
\begin{aligned}
    \frac{\partial L}{\partial p_{\theta, 1}} &= -\frac{p_1}{p_{1, \theta}} + \frac{1 - p_1}{1 - p_{1, \theta}} \\
    \frac{\partial L}{\partial h} &= \frac{\partial L}{\partial p_{\theta, 1}} \frac{\partial p_{\theta, 1}}{\partial h} = \left[-\frac{p_1}{p_{1, \theta}} + \frac{1 - p_1}{1 - p_{1, \theta}} \right] \left[ p_{1, \theta} (1 - p_{1, \theta}) \right] = p_{1, \theta} - p_1
\end{aligned}
$$

可以看到使用 BCE 损失时，在 sigmoid 层前的梯度的形式，是和 MSE Loss 时输出层的梯度的形式是一样的。
梯度难以传递是出现在，用 MSE Loss 训练一个最后一层是 sigmoid 而非直接 Linear 的网络。

在多分类场景下，分析是类似的，留待后面补充。

同时也可以看到，我们转换 MSE Loss 的这套方法，本质上可以用于转换很多 Loss，把大部分判别模型都可以看成是（条件于 $x$，关于标签/回归量 $y$） 的条件生成模型的拟合。


