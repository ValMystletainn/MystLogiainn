---
title: Normalizing Flow
keywords: "generative model"
---

## 引论

在上一篇文章中，我们介绍了自回归模型，说明了它是一个拆分超大规模数据分布为各种条件分布之积的方法。
并且强调了，虽然自回归最常用于构建文本等离散数据的生成模型，但它也可以用于构建图像等连续数据的生成模型。
但上篇文章的实际案例，我们给到的还是文本这类离散数据的生成模型。

这其实是因为即使是一维的连续随机变量的分布，它也比离散随机变量的分布表达起来要复杂得多（下文马上细说什么叫表达起来复杂）。
在深度生成模型的这一学术话题下，其实有很多种表示和拟合连续随机变量分布的方案。

本篇文章就来用来简要分析连续随机变量分布表达更难的地方，并引出本系列要介绍的第一个随机变量生成模型，Normalizing Flow。
在开头值得提出的是，Normalizing Flow 比 VAE 和 GAN 这些支持连续随机变量的生成模型提出要晚，本篇文章优先介绍它是出于系列文章前后的逻辑链条来考虑的。
后续介绍别的生成模型时，也有可能为了引入的自然性而调整所介绍方法的先后顺序。


## 离散分布和连续分布的参数化方法比较

抽象地来说，**参数化**是用一套参数组来刻画你所想表达的对象（分布、曲线曲面、物理对象...）。
参数不一样，对象的具体属性就不一样。

具体到我们现在在意的问题，那需要描述的对象就是随机变量的分布。
要参数化随机变量的分布，其实是**使得任给一随机变量的可取值，配合上你选定的参数，能或直接或间接计算出这一可取值的概率密度/概率**，也即把 pdf/pmf 这个函数，参数化的表达出来。

### 离散分布参数化

对于可能取值为有限个的离散随机变量，很直接表达它分布的方法就是列表，把分布列写出来。
举例来说，若现有一随机变量 $X$，它可能取值为 $x_1, x_2, \cdots, n$，所需要列的表就是指向量 $\textbf{p} = [p_1, p_2,\dots p_n]^T$，其中 $p_i$ 表示 $X=i$ 的概率。

这样就完全刻画了随机变量 $X$ 的分布。
随机给一个 $X$ 的可能取值 $x$，直接在相应位置查找，把对应的 $p$ 排出来就行。
而同时，我们还可以看到，为了满足概率分布的要求，对参数组 $\textbf{p}$ 需要有约束

$$
p_i \geq 0, \sum_{i=1}^n p_i = 1
$$

更重要的是，我们可以看到，这样的参数化方法，完备地表达了所有可能的 $n$ 值离散随机变量的分布。
**完备**的意思是指，对于任意一个 $n$ 值离散随机变量分布，都可以找到一组参数 $\textbf{p}$ 来刻画它；同时，对于任意一组参数 $\textbf{p}$，都可以找到一个 $n$ 值离散随机变量分布与之对应。
听上去很数学很拗口很严谨，但验证起来很简单直接，因为我们就是把分布列给写出来并当成了向量，就是这样一个关系

$$
n \, 值离散随机变量分布 \leftrightarrow 分布列 \leftrightarrow 向量 \, \textbf{p}
$$

确定好参数化方案以后，很直接的操作就是让神经网络的输出当做这套参数。
比如，在语言模型里，通常我们可以看到它最后几层是先让网络输出一个 $n$ 维向量，然后过 softmax 函数，给出最终输出，这个最终输出就是在下一个词元的分布列，也是可以说是参数化了 $p_\theta(x_t|x_{<t})$。

如果只是在离散分布里，引入那么多术语，其实有点没必要和咬文嚼字。
以它为案例是为了在过度到连续分布情形时，能有一个更清晰的对比。

### 连续分布参数化尝试

当我们考虑对连续变量进行参数化时，就会发现情况迅速复杂了起来，上面的思路几乎不能复刻。

如果提取离散变量参数化的精髓，就是直接把要刻画的函数的自变量和对应的函数值列下来，自然就对应上了。
但你会发现，在连续变量的情形下，这根本行不通。
在离散变量的 pmf 中，只有有限个可能的点，列出可取的概率值就是一个有限 $n$ 维的向量。
但在连续变量的情形下，即使是 $\mathbb{R}$ 上的连续随机变量，甚至是长度很小的一个区间才有正概率密度的随机变量，那要列出的随机变量/自变量和其对应的概率密度值/函数值，就会有无穷多个。
这是不可能由有限维的向量来表示，并由网络来输出的。


这条路行不通以后，可能会给出的一个方案是，听起来**高斯分布**在连续变量分布下很常见，那要不就先假定分布是高斯分布，用高斯分布的均值向量和协方差来刻画吧。
这确实是一个参数化方案，但你会发现它并**不完备**。
因为此时，不管你的参数怎么动，所表达的分布都一定会是一个高斯分布，即一定是某个中心点高值，往外逐渐衰减的单峰的 pdf。
显然这不是所有的 pdf 都会有的形状。
也就是说，有些分布就不好被表达出来了。
而这样的不好表达，甚至在分布拟合的情形下会导致“完全错误”的拟合效果。

举一些具体的例子。
例子都在 $\mathbb{R}$ 上考虑，都把模型分布考虑为 $p_\theta(x) = \mathcal{N}(x; \mu_\theta, \sigma)$，
即认为模型分布是一个均值为 $\mu_\theta$，方差为 $\sigma$ 的高斯分布，其中 $\mu_\theta$ 一定是可调的模型直接输出的参数 $\sigma$ 是一给定值。（这里留一个伏笔，可以考虑为什么不把 $\sigma$ 也当成可调参数，应该会在讲到 GAN 时揭晓）。


我们用这样的模型分布，试图拟合如下两个真实分布：

1. $[-1, 1]$ 上的均匀分布 $U(-1, 1)$。记为 $p_1$
2. 在 $[-2, -1] \cup [1, 2]$ 上的均匀分布。记为 $p_2$

为用 $p_\theta$ 拟合 $p_1$ 或 $p_2$，我们就需要分别计算 $p_\theta$ 与它们的交叉熵，并以此为损失函数来做优化。
这里，我们假设网络表达能力足够强，且样本足够多，这样我们就可以忽略实际训练时蒙特卡洛采样带来的误差以及因为网络表达能力不够强而导致 $\mu_\theta$ 不能取到 $\mathbb{R}$ 上所有的值而带来的拟合误差。
这样能简化分析过程，并且分析得到的结论会是拟合效果的一个上限，如果上限都不够好，那实际用有限样本训练时也就不会太好了。
在这样的假设下，优化变量就是 $\mu_\theta$，优化目标就是（积分形式的）交叉熵，即分别做如下优化问题：

$$
\begin{aligned}
  & \min_{\mu_\theta \in \mathbb{R}} & \quad \int_{-\infty}^\infty -p_i(x) \log p_\theta(x) \mathrm{d}x \quad i = 1, 2 \\
\Leftrightarrow & \min_{\mu_\theta \in \mathbb{R}} & \quad \int_{p_i(x) > 0} \frac{(x - \mu_\theta)^2}{2\sigma^2} \mathrm{d}x \quad i = 1, 2 
\end{aligned}
$$

从第一行到第二行，是通过代入 $p_\theta, p_i$ 的具体形式，丢弃与 $\mu$ 无关不影响最优值点的项，并乘上合适的放缩倍率得到的。通过代入具体的 $p_1, p_2$，并对 $\mu$ 求偏导，不难得到，以上优化问题，无论对于 $p_1, p_2$ 最优解都是 $\mu_\theta = 0$，可视化如下图：


````{figure}
:label: fig-gaussian-fit
```{kroki}
:src: tikz
:alt: "Gaussian Fit goodcase vs badcase"
:align: center
:width: 80%

\documentclass[border=3.14mm]{standalone}
\usepackage{tikz}

\begin{document}
\begin{tikzpicture}[x=1cm, y=5cm, 
    declare function={
        p1(\x) = (\x>=-1 && \x<=1) ? 0.5 : 0;
        p2(\x) = ((\x>=-2 && \x<=-1) || (\x>=1 && \x<=2)) ? 0.5 : 0;
        gauss(\x,\mu,\sigma) = exp(-(\x-\mu)^2/(2*\sigma^2))/(\sigma*sqrt(2*pi));
    }]

    \draw[->] (-3.5,0) -- (3.5,0) node[right] {$x$};
    \draw[->] (0,-0.05) -- (0,0.65) node[above] {$p$};
    
    % p1
    \draw[blue, thick, samples=200, domain=-1.2:1.2] 
        plot (\x, {p1(\x)});
    
    % p_theta
    \draw[red, thick, samples=200, domain=-3.5:3.5, smooth] 
        plot (\x, {gauss(\x,0,1)});
    
    \begin{scope}[xshift=8cm]
        \draw[->] (-3.5,0) -- (3.5,0) node[right] {$x$};
        \draw[->] (0,-0.05) -- (0,0.65) node[above] {$p$};
        
        % p_theta
        \draw[red, thick, samples=200, domain=-3.5:3.5, smooth] 
            plot (\x, {gauss(\x,0,1)});

        % p2
        \draw[orange, thick, samples=500, domain=-2.5:2.5] 
            plot (\x, {p2(\x)});
        
        % legend
        \draw[red, thick] (2,0.83) -- (3.5,0.83) node[right,black] {$p_\theta$};
        \draw[blue, thick] (2,0.73) -- (3.5,0.73) node[right,black] {$p_1$};
        \draw[orange, thick] (2,0.63) -- (3.5,0.63) node[right,black] {$p_2$};
    \end{scope}
\end{tikzpicture}
\end{document}

```

$p_\theta$ 拟合 $p_1, p_2$ 的效果对比。
````

可以看到如果用 $p_\theta$ 来替代 $p_1$，不好的点只是会把更接近于 $0$ 的部分有以更高的概率取到，不满足均匀性，但在 $[-1,1]$ 之外的部分能采样到的概率还是很低的，用 $p_\theta$ 代替 $p_1$ 并进行采样，采样出的结果大概率还是在 $p_1$ 的支撑集上的。
但如果来到对 $p_2$ 的拟合，我们会发现拟合的情况就很糟糕了。
$p_\theta$ 的峰还是出现在 $0$ 附近，但是对于 $p_2$，$[-1,1]$ 这个区间采出样本的概率应该完完全全是 $0$。
这时强行用 $p_\theta$ 来代替 $p_2$，采样出的结果就大概率会是 $p_2$ 中根本不可能采样出来的数据，也就说明我们这样的分布拟合做得很差。

并且可以想象到 $p_2$ 这个例子是可以从一维直接迁移到高维的。
假设我们有个手写数字 $1$ 的图像分布，$1$ 的位置大致在中心，但有一定的随机移动。
如果强行用高维高斯分布来拟合这个分布，也就是用 MSE Loss 来重构数据集里 $1$ 的图像。
我们把每个 $1$ 都当成一个小的数据峰，最后 $p_\theta$ 的均值会坐落在所有数据峰的平均值。
在未平移的 $1$ 的边缘附近，有平移的图像这部分像素还是高亮的，而没有平移的图像的像素已经是暗了的，两者进行平均，最后会导致 $1$ 的边缘更长慢慢的渐变灰暗下去。
做更夸张的假设，如果数字 $1$ 里其实可以更细分为两类图，一类是 $1$ 所对应的高亮像素在偏左的位置，另一类是 $1$ 所对应的高亮像素在偏右的位置。
也即两个峰距离很远，那最后得到的结果大概会像两个数据峰的平均值，也就是一个亮度只有一半的 $11$ 的图像。

### 对比反思

在本系列第一篇文章中提到过一个观点，**采样是一种运算**，对于任意复杂的分布，想做到用计算机模拟采样，都还是要通过一组简单分布下的随机值结合上分布本身的参数/概率密度函数形式，进行运算，来得到想要的采样结果。
我们不妨画一个包括神经网络计算分布参数和采样运算的运算图，再来对比离散分布的通用参数化方案和我们刚刚失败的高斯分布参数化方案，如下图左侧两图所示：

````{figure}
:label: fig-calc-map-param-sample
```{kroki}
:src: tikz
:alt: "Gaussian Fit goodcase vs badcase"
:align: center
:width: 80%

\documentclass[border=3.14mm]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes, calc}

\begin{document}
\begin{tikzpicture}[
    % Global styles
    >=Stealth,
    node distance=1.2cm,
    every node/.style={font=\small},
    % Custom node styles
    input/.style={rounded rectangle, fill=blue!20, draw=blue!50, thick, minimum size=8mm},
    nn/.style={rectangle, fill=orange!20, draw=orange!50, thick, minimum size=8mm},
    output/.style={rectangle, fill=green!20, draw=green!50, thick, minimum size=8mm},
    random/.style={circle, fill=red!20, draw=red!50, thick, minimum size=8mm},
    sampler/.style={diamond, fill=violet!20, draw=violet!50, thick, minimum size=8mm},
    sampleout/.style={rectangle, fill=teal!20, draw=teal!50, thick, minimum size=8mm}
]

% ===================== LEFT PANEL: Discrete Sampling =====================
\begin{scope}[local bounding box=left]
    % Nodes
    \node[input] (input1) {$\mathbf{c}$};
    \node[nn, below=of input1] (nn1) {$\mathrm{NN}_\theta$};
    \node[output, below=of nn1] (out1) {$p_1,\dots,p_n$};
    \node[random, right=of out1] (rand1) {$\epsilon$};
    \node[sampler, below=of out1] (sample1) {Sample};
    \node[sampleout, below=of sample1] (result1) {$\mathbf{x}$};
    
    % Arrows
    \draw[->] (input1) -- (nn1);
    \draw[->] (nn1) -- (out1);
    \draw[->] (out1) -- (sample1);
    \draw[->] (rand1) -- (sample1);
    \draw[->] (sample1) -- (result1);
    
    % Title
    \node[above=0.2cm of input1] {Discrete Distribution};
\end{scope}

% ===================== CENTER PANEL: Gaussian Sampling =====================
\begin{scope}[local bounding box=center, xshift=5cm]
    % Nodes
    \node[input] (input2) {$\mathbf{c}$};
    \node[nn, below=of input2] (nn2) {$\mathrm{NN}_\theta$};
    \node[output, below=of nn2] (out2) {$\mu$};
    \node[random, right=of out2] (rand2) {$\epsilon$};
    \node[sampler, below=of out2] (sample2) {Sample};
    \node[sampleout, below=of sample2] (result2) {$\mathbf{x}$};
    
    % Arrows
    \draw[->] (input2) -- (nn2);
    \draw[->] (nn2) -- (out2);
    \draw[->] (out2) -- (sample2);
    \draw[->] (rand2) -- (sample2);
    \draw[->] (sample2) -- (result2);
    
    % Title
    \node[above=0.2cm of input2] {Gaussian Distribution};
\end{scope}

% ===================== RIGHT PANEL: Continuous Generation =====================
\begin{scope}[local bounding box=right, xshift=10cm]
    % Nodes
    \node[input] (input3) {$\mathbf{c}$};
    \node[random, right=of input3] (rand3) {$\epsilon$};
    \node[nn, below=1.5cm of $(input3.south)!0.5!(rand3.south)$] (nn3) {$\mathrm{NN}_\theta$};
    \node[sampleout, below=of nn3] (result3) {$\mathbf{x}$};
    
    % Arrows
    \draw[->] (input3) -- (nn3);
    \draw[->] (rand3) -- (nn3);
    \draw[->] (nn3) -- (result3);
    
    % Title
    \node[above=0.2cm of input3] {$\quad \quad$ Deep Continuous Distribution};
\end{scope}

% ===================== DIVIDING LINES =====================
\draw[dashed] ($(left.north east)+(0.5,0)$) -- ($(left.south east)+(0.5,0)$);
\draw[dashed] ($(center.north east)+(0.5,0)$) -- ($(center.south east)+(0.5,0)$);

\end{tikzpicture}
\end{document}

```

不同分布参数化方案以其采样运算的计算图
````

从函数本身的复杂度看，离散分布所对应的 pmf 是一个只有 $n$ 种自变量可能性的函数，而连续分布所对应的 pdf 则是一个有不可列无穷种自变量可能性的函数，明显后者更复杂。
而参数化后，离散分布同样得到了 $n$ 个参数，然后再进行采样，而更复杂的连续分布则只得到了一个参数/参数向量，然后参与采样。
最后一步的采样即不是什么复杂的运算，也得不到很多维的复杂输入，自然最后就产生不了复杂的分布，做不出多峰的样子。

既然如此，那我们可不可以考虑把采样用的熵源 $\epsilon$ 从最后一层的采样头移动到整个网络的最前端。
因为神经网络本身是多层的，多个函数的复合运算，把 $\epsilon$ 移到开头就更可能等效地能得到一个复杂的分布，只不过这时候我们不再有分布 $p_\theta$ 的显式参数化表达了，而是被网络参数 $\theta$ 隐式地参数化，并且采样也被融进了整个网络里，直接从网络拿到的就是采样的结果。

这样把熵源在网络很早期就进行输入的方案，就是构建由神经网络决定的连续型随机变量分布的通性方案。
但是这样构造以后，$p_\theta$ 不再能显式得到，交叉熵不好计算，为解决这个问题，就会产生我们在这篇文章里要介绍的 Normalizing Flow 模型，以及之后会介绍的 VAE, GAN, flow mathcing, diffusion, ...

## 增添约束

现在我们来分析一下，为什么把熵源 $\epsilon$ 放到网络起始输入后 $p_\theta$ 就难以显式得到了。
对于最开始的随机值 $\epsilon$，因为它是从简单的分布（如标准高斯、均匀分布等）采样出来的，所以知道 $\epsilon$ 代入相应的 pdf 就能算出它对应的似然值。
但当经过一层网络计算后，假设得到的值叫 $h_1$，因为它是由 $x$ 和有随机性的 $\epsilon$ 共同计算得到的，所以其实 $h_1$ 也是一个随机变量。
但 $h_1$ 的概率密度值就很难得到了，一定要按照定义去计算，需要找到 $h_1$ 所对应的逆像，也即所有能配合 $x$ 得到 $h_1$ 的 $\epsilon$ 的集合，累计这些 $\epsilon$ 的概率值（并按照局部变换的放缩比乘倍率），才能得到 $h_1$ 的概率密度值。
对任意一个映射找逆像是一个计算开销很大的事（枚举输入判断是否等于结果），而这还仅仅是一层网络的计算，对于多层网络就更不能接受了。

所以我们考虑给网络增加一些约束，用来解决似然值计算的问题。
我们希望网络所对应的映射 $f_\theta$ 满足以下条件：

1. $f_\theta$ 可逆。
2. （可选）$\det \frac{\partial f}{\partial \epsilon}$ 便于计算。

从而我们可以利用如下的公式，来计算 $p_\theta(x)$，然后能算出交叉熵 $\mathbb{E}_{x \sim p} \left[-\log p_\theta(x) \right]$ ，用其作为损失函数训练网络。

$$
p_\theta(x) =  p_\epsilon(\epsilon) \left| \det \frac{\partial f^{-1}}{\partial x} \right| = p_\epsilon(\epsilon) \left| \det \frac{\partial f_\theta}{\partial \epsilon} \right|^{-1}
$$

这个公式也有非常符合物理直觉的解释和简要推导。
因为 $f_\theta$ 可逆，所以对于任意的 $x$ 我们自然能找到唯一的 $\epsilon$ 使得 $f_\theta(\epsilon) = x$。
对于这样的一对 $x, \epsilon$，取 $x$ 附近的小邻域 $\mathrm{d}x$，它自然也会通过 $f_\theta$ 的逆映射对应得到 $\epsilon$ 周围的小邻域 $\mathrm{d}\epsilon$，并且因为 $f$ 可逆，抽出的 $x$ 落在邻域 $\mathrm{d}x$ 的概率就是抽 $\epsilon$ 时它落在邻域 $\mathrm{d}\epsilon$ 的概率，**得到方程**: $p_\theta(x) \mathrm{d} x = p_\epsilon(\epsilon) \mathrm{d} \epsilon$。为表达出 $p_\theta(x)$ 就需要把微元 $\mathrm{d}x$ 移过去，这是一个体积微元，移过去是做 $\mathrm{d}\epsilon$ 和 $\mathrm{d} x$ 的体积之比。对一个线性映射，映射前后体积比就是线性映射的行列式的绝对值，在原始映射里，因为我们考虑的是 $x, \epsilon$ 处的微元，也就是同时做了这映射 $f^{-1}$ 的局部线性化/微分，故 $\mathrm{d}\epsilon$ 和 $\mathrm{d} x$ 的体积比就是 $\left| \det \frac{\partial f^{-1}}{\partial x} \right|$。

但构造一个可逆的神经网络并非易事，这里给出 [NICE](https://doi.org/10.48550/arXiv.1410.8516) 和 [RealNVP](https://doi.org/10.48550/arXiv.1605.08803) 两篇文章为满足可逆性而进行的模块构造的基本思路：

对于输入 $\epsilon$，可以把它按维度任意划分为两个部分 $\epsilon_1, \epsilon_2$，然后以如下运算得到下一层的输出 $h = [h_1, h_2]$

$$
\begin{aligned}
    h_1 &= \epsilon_1 \\
    h_2 &= m_\theta(\epsilon_1) + \epsilon_2
\end{aligned}
$$

其中 $m_\theta$ 是任意的带参数 $\theta$ 的神经网络常用运算。
如果需要拟合的是分布是条件分布，条件 $c$ 也可以在此一并输入给 $m_\theta$。
通过移项解方程，不难确认以上从 $\epsilon \to h$ 的运算的**可逆性**，即尝试用 $h_1, h_2$ 表示 $\epsilon_1, \epsilon_2$：

$$
\begin{aligned}
    \epsilon_1 &= h_1 \\
    \epsilon_2 &= - m_\theta(h_1) + h_2
\end{aligned}
$$

同时它也满足了之前提到的约束条件 2，变换前后的雅可比矩阵的行列式便于计算，从上二式不难得到

$$
\begin{aligned}
    \frac{\partial h}{ \partial \epsilon } &= \left[
        \begin{matrix}
            I_1 & 0 \\
            \frac{\partial m}{\partial \theta_1} & I_2
        \end{matrix}
    \right] \\
    \frac{\partial \epsilon}{ \partial h } &= \left[
        \begin{matrix}
            I_1 & 0 \\
            -\frac{\partial m}{\partial h_1} & I_2
        \end{matrix}
    \right] \\
\end{aligned}
$$

都是下三角行列式，其行列式的值都为 $\det I_1 \cdot \det I_2$ 也就都是 $1$。
不需要再用自动求导包来数值计算雅可比矩阵和其行列式的值，减小了计算样本似然值时的计算量，变成只需要代入逆运算得到 $\epsilon$ 以及从 $p_\epsilon$ 的公式中得到似然值即可。

## 增添层数

正如常规的神经网络需要堆叠多层，来增加网络的表达能力，上面构建的可逆映射模块，也通常是需要堆叠多层来构成最终的可逆神经网络 $f_\theta$。
若每一个中间层输出如果记为 $h^{(1)}, h^{(2)}, \dots h^{(l)}$，在生成模型中我们还会关注它们对应的似然值 $p_{\theta, h^{(1)}}(h^{(1)}), p_{\theta, h^{(2)}}(h^{(2)}), \dots, p_{\theta, h^{(l)}}(h^{(l)}) $。
经过网络的逐层变换，样本层面上是 $\epsilon \to h^{(1)} \to \cdots \to x$，而分布层面上是 $p_\epsilon \to p_{\theta, h^{(1)}} \to \cdots \to p_\theta$。
一个个样本点像粒子，从 $\epsilon$ 运动变成 $x$；而整个分布像由粒子堆积成的液体，随着变换进行发生流动，从简单的 $p_\epsilon$ 流动或者说被塑造成了复杂的 $p_\theta$，这也正是这类方法的名字 Normalizing Flow 中 **Flow** 的由来。


值得一提的是，相比于普通神经网络，对于这类网络做叠层的时，有以下要额外注意的点：

### 通道排列更换

如果说经过每层映射时，上半部分 $h_1$ 和下半部分 $h_2$ 的通道编码都不更换，不难看出

$$
\epsilon_1 = h_1^{(1)} = h_1^{(2)} = \cdots = h_1^{(l)} = x_1
$$

也就是说上半部分通道的内容完全没有发生变化，这显然是不合理的。
解决的办法也很简单，在每进一个新的网络层时，把上半通道的地位和下半通道的地位互换即可。即：

$$
\begin{aligned}
    h_1^{(1)} &= \epsilon_1 \\
    h_2^{(1)} &= m^{(1)}_\theta(\epsilon_1) + \epsilon_2 \\
    h_1^{(2)} &= h_1^{(1)} + m^{(2)}_\theta(h_2^{(1)})\\
    h_2^{(2)} &= 0 + h_2^{(1)} \\
    \vdots \\
\end{aligned}
$$

### 通道放缩

为了满足必要的可逆性假设，我们就必须使用和生成样本等维度的 $\epsilon$ 来做输入，这已经比较维度冗余了。
又因为我们的运算构造保证了每层运算的雅可比行列式是 $1$，每层运算之后每维度的特征的收缩膨胀率都是 $1$，也不能让我们丢掉冗余的特征。
而如果稍微对运算的构造进行放松，保证是雅可比行列式是下三角块，且主对角上的两矩阵块还是对角阵，那么雅可比行列式的值就是对角线元素之积，$n$ 个元素之积，还是很好算的，同时也让我们有机会让每维度的特征的收缩膨胀率不都是 $1$。

比如将层的形式换为：

$$
\begin{aligned}
    h_1 &= \epsilon_1 \\
    h_2 &= m_\theta(\epsilon_1) + \epsilon_2 \odot \exp(s(\epsilon_1))
\end{aligned}
$$

其中 $s$ 是一个 $\mathbb{R}^{n/2} \to \mathbb{R}^{n/2}$ 的函数，取 $\exp$ 主要是为了保证正放缩倍率，以及计算 loss 时递推逐层的对数似然可以和 $\log$ 对消。
容易验证，上式还是可逆的，可以用 $\epsilon_1, \epsilon_2$ 显式表达 $h_1, h_2$；并且上式的雅可比行列式是 $\Pi_{i=1}^{n/2} \exp(s_i(\epsilon_1)) $ 不再是常数 $1$，也就是说对每层中对重要或不重要的特征有了一定的选择性放缩。

在 NICE 和 RealNVP 的原文中，已经对 MNIST 等数据集上的样本生成做了简单尝试。
而 [Glow](https://doi.org/10.48550/arXiv.1807.03039) 等工作尝试构造了更复杂的可逆变换层，做了更复杂的图像生成任务。

## 微分方程化

如果要提炼上述构造可逆变换层的精髓，就是通过一般的神经网络层，组合上一定的特殊设计过的运算，来达到一层为单位整体是可逆的效果。
另一种思考思路来自 [NeuralODE](https://doi.org/10.48550/arXiv.1806.07366)。
如果考虑一个 $\mathbb{R}^d \to \mathbb{R}^d$ 的普通神经网络层 $m_\theta$，我们将它的输出认为是一种速度/速度场，用它来移动样本，即考虑如下微分方程：

$$
\frac{\mathrm{d} x}{\mathrm{d} t} = m_\theta(x), \quad \text{when } t = 0, x = \epsilon
$$

这样随着微分方程隐式定义的变换走到终止时间 $T$（不失一般性，可设 $T = 1$），就完成了又神经网络定义的从 $\epsilon \to x$ 的变换。
同时呢，从微分方程所描述的这个物理过程，不难验证其**可逆性**，对于任意给定的 $x$，我们将其输入给网络，得到速度场 $m_\theta(x)$，逆着速度线方向 $-m_\theta(x)$ 移动，时间从 $1$ 推到 $0$，每一步都会和从 $\epsilon \to x$ 的轨迹重合，最终会得到这个 $x$ 相应的 $\epsilon$，也就说明了由 $m_\theta$ 和上述 ODE 所定义的 $\epsilon \to x$ 的变换的可逆性。

为了构造更复杂的变换，一般考虑时变的速度场 $m_\theta(x, t)$，也就对应在网络层多增加一维 $t$ 的输入即可，上述可逆性依然可验证成立。

接下来就该推理在这个 ODE 模式下的似然值计算了。
但 NeuralODE 并不完全是关注于生成模型的一篇文章，其叙事主线是残差连接是一种差分方程，将其连续化为微分方程后能不能得到新的好的网络设计，它原文关于似然值计算的推导部分比较新手不友善。
另外 NeuralODE 的这套思路会和 Flow Matching 有一定的联系，所以关于这样定义 $\epsilon \to x$ 的变换后如何做算出似然值，做交叉熵，训练网络，就留待 Flow Matching 那一节再做讲解。


## 总结

在本篇文章中，我们开始从离散分布的拟合过度到连续分布的拟合的方法介绍。
首先分析了连续型多峰随机变量的分布拟合的困难，然后指出需要更早的将熵源输入给更复杂多层的神经网络，来达成将复杂多峰分布纳入 $p_\theta$ 可行域的目的。
随后为了解决样本似然值的计算问题，引入了 normalizing flow，它是一套基于可逆神经网络映射的生成模型。
以 NICE 和 RealNVP 等工作为展开，说明了如何构造可逆的神经网络层，和具体计算样本似然值并进行交叉熵损失训练。
同时也引入了 NerualODE 所提出的连续可逆流，为之后介绍 flow matching 埋下伏笔。
