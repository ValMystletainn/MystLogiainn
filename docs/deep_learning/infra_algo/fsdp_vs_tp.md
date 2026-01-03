---
title: FSDP 对比 TP
keywords: parallelism
subtitle: 一次从llm与dit基本并行方案的选型差异的好奇探索
---

## 背景

我自己的主业工作是图像/视频生成算法工程师，在这块主流的模型侧的并行方案就是 pytorch 原生的 FSDP。
当然我也知道，大部分做 LLM/VLM 的同学用的是 megatron 为主的开发框架，主要的模型侧的拆分并行策略就变成了 TP PP。
这两种选型都能在各家的 DiT/LLV/VLM 技术报告中有体现，当时，我对此的理解是，旗舰 DiT 比旗舰的 LLM/VLM 还是参数量太小了，而且 scale 上去的收益不像人家那么明显且可预期，没有进一步做大的需求（指能总结出 scaling law 那种比较定量的结果），所以两者的主流并行方案有差异。

但得到这样的解释后就放过自己，其实有点不求甚解。
为什么说模型更大了以后 FSDP 就不如 TP 了呢？具体是哪里发生了问题呢？
直到最近，偶然在知乎的推荐下刷到了 [AsyncHZP](https://doi.org/10.48550/2510.20111) 这篇文章的推广，发现 LLM 的场景下，人家也用 FSDP like 的主策略实现了 sota 的 MFU。
这让我重新思考了这个问题， FSDP 和 TP 两种并行策略应该如何对比，分别的优劣势是什么呢？
这一段时间的搜索查阅后，我认为我能更好的解答这个问题了，所以我尝试从一个算法工程师，也即并行策略的使用者的角度来再次回答这个问题。

## 两种策略的并行流程、通信量分析、通算遮掩实现

在我能搜到的 TP 和 FSDP/zero 系列的解读文章中，都基本完整的描述了并行策略和通信量分析，但只有很少数的文章描述了通算遮掩实现的情况。
而在这次调研中，我认为通算遮掩的实现条件，是这两种策略的一个很重要的对比点，所以我会将其补充在接下来对这两种并行策略的基本描述中。

### FSDP

这里主要描述 FSDP2 的流程, FSDP1 略有差异，但不影响大部分的分析和结论。

![fsdp_flow](https://docs.pytorch.org/tutorials/_images/fsdp_workflow.png)

只看图有些抽象，我们这里用文字再把过程描述一遍 （忽略掉 cpu offload 相关的部分）：

- 用 FSDP2 包裹模型后，模型会把 **每层** 的各种参数都切分 N 份，每张卡上只常态化保留自己的 $1/N$ 份参数。从而在每张卡上，维护的优化器状态，也只用常态化保留对应的 $1/N$ 参数的优化器状态。
- 在每次 forwad 时，每个子层计算时，由于需要完整的参数才能进行把计算跑通，所以要把切片找齐，做一次 `all-gather` 让所有卡，都拥有 **当前层** 的全量参数，从而让计算能跑通，而跑完之后，因为要节省显存，再把这层的参数中，不属于自己的那 $N-1/N$ 份参数释放掉
- 同样在 backward 的时候，对于当前的子层，要有全量的参数，才能计算梯度，所以也需要 `all-gather` 让所有卡都拥有 **当前层** 的全量参数，而跑完之后，计算得到梯度，再释放掉不属于自己的那 $N-1/N$ 份参数，并做 `reduce-scatter` 来得到自己那 $1/N$ 个参数的 dp 平均后的梯度。
- 上述描述是顺序的，看起来像是，先做层0的通信，再做层0的计算，再释放层0的参数，再做层1的通信，再做层1的计算，再释放层1的参数，...。但不难发现，这里是可以做 **预读取** 的，来遮掩掉通信的，即：在做第 $n$ 层的计算的时候，就开始预先读取 $n + 1$ 层的参数。这也是 fsdp2 的默认配置(即 `forward_prefetch`,  `backward_prefetch`)

所以对于纯 FSDP 来说。由于每层的参数都完完整整的切了 $N$ 份，对于参数量为 $\phi$ 的模型的以 fp32 做 gradient reduce 的 adam 混精度训练来说，本应该由模型fp32主权重、模型 bf16 副权重、模型 fp32 梯度，优化器状态所占用的 $18P$ 显存，在 FSDP 的 $N$ 份切分后，就变成了 $\frac{18\phi}{N}$。当然，实际运行中，从上面可以看到模型相关峰值显存应该比这稍微多一些些，因为实际计算中，最极限会存在两个层的全参数。

而通信量也很好计算，跟着 forward 的流程走，每张卡会逐层的把自己的每层参数，以副权重的精度发 $N-1$ 份，然后收 $N-1$ 份，层层递进，相当于把整个模型的切片后都差不多收发了一遍，即每张卡有 $\frac{N - 1}{N} \times 2 \phi$ 的发送量。backward 时也是类似，对权重有 $\frac{N - 1}{N} \times 2 \phi$ 的发送量，同时还考虑梯度的 reduce scatter，虽然以 fp32 做 gradient reduce，但这只在做加减时产生，通信时还是以原生计算出的 bf16 精度的梯度来通信，所以对梯度也有 $\frac{N - 1}{N} \times 2 \phi$ 的发送量。当然如果从全局视角考虑，一般我们工作在双工信道，一张卡能同时按照标称速率来同时收发，所以计算全体的通信量时，就变成了 $\text{send}_{\text{single}} * \frac{N}{2}$ 而非直接乘 $N$。

### TP

对于 TP，它利用的是矩阵乘法的可分块性，直接将各种线性层的权重进行分块，将不同分块放置于不同的卡上进行计算，并适时地汇总计算结果。由于权重切分了，每张卡上模型参数占用的显存就变小了，以此来达成减小每张卡上模型占用的显存的目的。

这里为了描述方便，使用 FFN 层的 TP 来展示，对于 attention 层，因为 TP 只涉及前后两个线性层的切分，对中间的 attention 计算是无感的，所以可以完全类比。

![tp_mlp_flow](https://ar5iv.labs.arxiv.org/html/1909.08053/assets/mlp_mp_2.png)

上图是一个 TP 2 的例子， TP N 也类似。对于一个 FFN 层，第一个 线性层的权重 $A$ 采用 column wise 的切分，$A = [A_1, A_2]$ 从而根据 $Y = XA = [XA_1, XA_2]$ 可以直接把对 $X$ 在两张卡上分别与 $A_1$, $A_2$ 做矩阵乘法，得到分片的结果 $[XA_1, XA_2]$ 并分别保留在各自的卡上（有需要时再通信组合出完整的 $Y$）。

后一步是激活函数，是一个 elementwise 的计算，不需要完整的结果 $Y$，还是每张卡用分片 $Y_1, Y_2$ 各自计算就好。

然后对于第二个线性层权重 $B$, 我们采用 row wise 的切分 $B = [B_1^\top, B_2\top]^\top$，对最终结果 $Z = YB = [Y_1, Y_2] [B_1^\top, B_2^\top]^\top = Y_1B_1 + Y_2B_2$。
可以看到，这样列行交错的切分，使得我们在每张卡上各自都拥有自己的 $Y_1$ 和 $Y_2$，可以很方便的跟卡上的对应分块 $B_1, B_2$ 做大块的矩阵乘法，到最终实在不行了，才需要做卡之间的 `all-reduce`，完成那个加法。
即在 forward 的最后，才会产生通信，且这个通信是和激活值相关的，batch size 越大 / sequence 越长通信量越大。假设激活值 $X$ 的大小为 $\alpha$，那么 TP $N$ 时，到最后这个 `all-reduce` 的通信量就为 $2\frac{\alpha}{N} N = 2\alpha$

backward 时同理，因为 backward 求梯度时会用各种转置后的矩阵乘法，从这个核心结论出发，简单推导一下，会在计算 $dX$ 时，因为 $dX = dYA^\top = dY_1A_1^\top + dY_2A_2^\top$ 才需要通信，通信量也同上。

而 TP 如何去实现通算遮掩呢？虽然我们公式写的是 分块里做矩阵乘法 $Y_1B_1$ 但实际上一个大的矩阵乘法，在**每一张卡内**的底层计算时，也是通过分块，逐步喂到 tensorcore 中来完成计算，所以我们完全可以把先计算好的分块，先做它的 `all-reduce`，同时计算下一个分块的矩阵乘法... 这样来完成通算遮掩。

在 transformer 中，各种线性层就是参数的主要来源，norm layer 等占用的参数量相对较少，用 TP N 来切分模型时，近似可以认为每张卡上只拿到了 $\frac{1}{N}$ 的权重，对应的主权重大小，optimizer state 大小也就减小到了 $\frac{1}{N}$，从而也达成了在每张卡上节省到 $\frac{1}{N}$ 的模型权重相关的显存占用的目的。

## 定量分析 - 通信量对比（伪对比）

上面简单描述了 FSDP 和 TP 的通信量，以 forward 单层 transformer block 为例进行比较，以下简记 $d$ 为 hidden size 大小， $l$ 为序列总长度。

，TP 的通信量是 $2\alpha * 2$，其中 $\alpha$ 为单层的激活值的数据所占用显存大小，常规的bf16混精度训练下，即 $\alpha = 2ld$ ，后面的系数 $2$ 是指在 attention 和 mlp 后的 linear 各有一次 `all-reduce`；那么单层 forward 的总通信量就是 $8ld$

FSDP 的通信量是 $\frac{N - 1}{N} \times 2 \Phi \approx 2\Phi$，其中 $\Phi$ 为单层的参数大小，一个不开 GQA，mlp ratio 为 $4$ 的 vannila transformer block 中 $\Phi = 3d^2 + d^2 + 4d^2 + 4d^2 = 12d^2$（四项分别为 qkv 投影, attention out 投影，mlp fc1 投影，mlp fc2 投影），那么单层 forward 的总通信量就是 $24d^2$。（实际上 llm 中已经是 swiglu 主导了， mlp ratio 会发生一些变化，导致下面的比例变化）

简单做一下比较，可以知道 $l = 3d$ 是 FSDP 和 TP 的通信量分野瓶颈，$l > 3d$ 时，序列比较长，激活值变多，TP 中由激活值主导的通信量就更多了，FSDP 通信量更少，而$l < 3d$ 序列较短时，激活值较少，FSDP 中由权重决定的通信量就显得更多了。

以 qwen2.5 为例（这里刻意避开 qwen3，后面就会知道为什么），3B, 7B, 14B, 32B, 72B 尺寸的模型分别的 hidden size 是 2048, 3584, 5120, 5120, 8192，那即使是最小的 3B 模型，临界 sequence length 都是 6144 了，对于稍大的模型得到万数量级，对于 LLM 中，除了长文本和 RL 的大部分训练负载，是不会有那么长的序列长度的，所以大家都以 TP 为主。

而对于 DiT 来说，即使是比较大的 flux1，flux2，hidden size 也就 3072,6144，同样，临界长度也在万量级，但是一张 1k 的图 token 长就在 4k~5k 了，万量级的临界长度也就四五张图，而单样本序列长度更长了，一个样本就能轻松打穿，甚至还需要 SP 的额外支持，所以大家就主用 FSDP。

以上分析看似很合理，并符合现实地解释了 LLM 和 DiT 的主并行策略的选型差异。
很长一段时间我也是这么认为的，但其实也有很多疑点，比如说这里的 $l$ 是 total sequence length，那完全可以通过 packing 更多的样本来增长单卡上的序列长，毕竟 LLM 的训练的总 token 数还是很大的，好像用 FSDP 也不是不行；看起来好像 TP 在大部分场景更优通信量更小，但都说 TP 不能跨机，而 FSDP 在视频模型里跨个位数的机还是能用的，而且效率损失也不明显...，直到刷到了 AsyncHZP 的推广文章，发现 LLM 上用 FSDP 也是能达到 SOTA 的，又有了这一轮新的探索。

## 定量分析 - 达成完美通算遮掩的条件

结论先行，**比较通信量的意义不大，重要的是比较并行策略实现完全通算遮掩的条件**。

FSDP 和 TP 的通信量有差异不假，但更重要的事是与它们的通信试图发生 overlap 的计算的范围不一样。对于 FSDP，有整层的计算与下层的通信来做 overlap，对于 TP，只有当前的切分后的矩阵乘法和当前矩阵乘法的通信做 overlap。
通信量 general 较大的 FSDP，能 overlap 的计算范围也更广，只要在计算完成前完成这样更大的通信，就不会降低训练效率，甚至某种意义上来说 “充分利用了系统的闲置的通信性能”。

所以接下来就详细分析一下两种并行策略达成完全通算遮掩的条件。
一般性的分析思路是，计算每个通算遮掩部分的计算量和通信量，分别除计算效率和通讯速度得到计算时间 $T_\text{compute}$ 和通信时间 $T_\text{communication}$，则当 $T_\text{compute} > T_\text{communication}$ 时，可以认为通信量已经完全被计算遮掩了，并行策略引入的额外通信不会影响训练效率。


以下分析仅分析 forwad。
并同时还是沿用上面的记号，记输入序列的总长为 $l$ 模型 hidden size 为 $d$，mlp ratio 假设为 $4$，投影层无 bias 采用 MHA，忽略 norm layer 的计算和参数。
对于其他变体的模型，以及 backward 过程，可以类比以下分析思路。

### FSDP

前面定性分析时已经提到， FSDP 的通算遮掩，是在计算当前 block 时， prefetch 下一个 block 的参数。
那自然地，通信量则近似是单层中所有投影矩阵所占的数据大小 $2\Phi = 2(3d^2 + d^2 + 4d^2 + 4d^2) = 24 d^2$，而计算量就是一层的 FLOPs。

一个 transformer block 里主要的计算有上面这四个 qkv 投影，attention out 投影，mlp fc1 投影，mlp fc2 投影，还有 attention 部分的计算。
这里忽略掉 attention 部分计算产生的 FLOPs，它跟单个序列的长度关联比较大，在当前的记号下是波动的。
这样忽略之后，我们则会得到一个偏低的计算 FLOPs 导出一个偏低 $T_\text{compute}$，若这样还能达成 $T_\text{compute} > T_\text{communication}$，则在实际的运行中，由于计算时间只会比理论的 $T_\text{compute}$ 更大，而通信量通信时间还是不变，从而更能达到计算遮掩通信的效果。

一个 $M\times K$ 的矩阵和 $K \times N$ 的矩阵做矩阵乘法的 FLOPs 是 $2MKN$，从而在实际的 block 中，投影相关的 FLOPs 是 $2ld3d + 2ldd + 2ld4d + 2l4dd = 12ld^2$。

记单卡标称 FLOPS 是 $C$，rank 间的通信带宽是 $W$，则可得到

$$
\begin{aligned}
T_\text{compute} &= \frac{12ld^2}{C} \\
T_\text{communication} &= \frac{24d^2}{W}
\end{aligned}
$$

目标是 $T_\text{compute} > T_\text{communication}$，简单对不等式移项后可得等价的条件为

$$
\frac{l}{2} > \frac{C}{W}
$$

这样化简是为了把 $\frac{C}{W}$ 放在一起，它是一个稍微有物理意义的值，表明了计算卡和它对应的通信信道能提供的计算通信强度阈值。
找一些典型值带入，对 A100 和 H100，$C$ 就用 datasheet 上的 FLOPS，分别是$312\times10^{12}, 989\times10^{12}$，如果通信走 nvlink，那通信带宽 $W$ 就是 $600\times10^{9} \text{B/s}, 900\times10^{9} \text{B/s}$，从而得到 $\frac{C}{W}$ 的关键值是 $520, 1098$，也就是总长度在 $1000,2000$ 量级时，仅节点内的做 shard 的 FSDP （或者用 pytorch 官方说法叫 HSDP） 就不会是瓶颈。
即使用 MMDiT 这种 FLOPs 不变，参数量翻倍的结构，把通信量范围，也只会把关键长度提升到 2000, 4000，不太会是瓶颈。
即使是 shard 跨了节点，按照 $100 \text{Gb/s} = 12.5 \text{GB/s}$ 的通信速率来看，在 $l$ 有万到十万量级的序列时，也能做到完全通算遮掩，在视频场景下大概可以做到。

### TP

在 TP forward 中，通算遮掩只能发生在 row wise tp 的矩阵乘法时，通过边算矩阵乘法的小分块边通信已经算好的结果进行 all reduce，达成通算遮掩。

这个类矩阵乘法有两个发生的位置，attention 的 projection out，和 mlp 的 fc2 投影。二者乘法时的 input 维不太一样，但是 output 维都是 $d$，为统一分析，我们记为先记 input 维维 $d_{\text{in}}$，而 TP 的切分为 $N$ 片。
从而在 forward 时，一个原始为 $d_{\text{in}} \to d$ 的矩阵乘法， 在 tp 切分后，每个 rank 上应该是 $\frac{d_{\text{in}}}{N} \to d$ 的矩阵乘法。
则其 FLOPs 是 $2l\frac{d_{\text{in}}}{N} d$。
而其通信量产生在最后的 的 all reduce，通信量为 $2ld$，同理可得到

$$
\begin{aligned}
T_\text{compute} &= \frac{2l\frac{d_{\text{in}}}{N} d}{C} \\
T_\text{communication} &= \frac{2ld}{W}
\end{aligned}
$$

目标是 $T_\text{compute} > T_\text{communication}$，简单对不等式移项后可得等价的条件为

$$
\frac{d_\text{in}}{N} > \frac{C}{W}
$$

对于 attention out projection， $d_\text{in} = d$ 而对于 mlp fc2 projection， $d_\text{in} = 4d$，考虑最极限的条件，attention out projection 会首先产生瓶颈，即需要满足 $\frac{d}{N} > \frac{C}{W}$。

同样，带入 A100，H100 的典型值可以得到切分后的关键 hidden size 为$520, 1098$，不取太激进的 TP 值，一般都能满足要求。
而如果 TP 要跨节点，则关键 hidden size 会变到万量级，几乎没有模型能达到这个配置。

同时这里对比 FSDP 可以看到，因为 TP 和矩阵乘法本身的分块性绑定得比较强烈，通信是不能提前或者推后发生的，不能跟更大范围的计算产生 overlap。


## 其他比较

到这里，我们可以得出结论是 FSDP/TP 都是很好的算法，在恰当配置下都能在 DiT/LLM 的场景下达成通算遮掩，不给训练带来额外的通信开销。
这解释了 AsyncHZP 这篇文章的效果，即 LLM 场景下用 FSDP 为主的策略也可以达到 sota MFU。
但还是不能解释为什么 DiT 常用 FSDP 而 LLM 常考虑 TP，难道仅仅是习惯问题？codebase 有了不想改？这是一方面因素，但还有更深层次的对比可说。

### FSDP 的额外优点

FSDP 的额外优点是，模型无关，侵入性不强。
这对于 DiT 来说是一个很大的优势。目前各家的 DiT 都有或多或少不一样的模型结构，还有很多的探索空间，FSDP 这种模型无关的并行方式，可以很方便大家做结构上的探索。

另外，像 bagel 等这样的 unified model，以及纯 LLM 的 gpt-oss, qwen3 next，这些架构很新颖的模型，使用 FSDP 支持起来都很快，再次体现了模型无关侵入性不强的优势。

另一方面，像 RL 这样计算流程比较复杂的训练过程（LLM 和 DiT RL 都是），使用 FSDP 为主的框架，开发起来也很便捷，比如现在能找到的各种 DiT RL 的 repo，LLM RL侧，一开始的 verl 还有最近 sglang & slime 猛猛更新的 FSDP like feature。

最后，相比 TP 来说， FSDP 更好与其他并行策略正交，比如视频模型里一般就会在同时使用 usp，或者 ring attention 来支持长序列。
而主推 TP 的 megatron 用的是与之耦合的 CP，但是实际使用时可能要对序列最开始就做一些 padding 处理。

### TP 的额外优点

TP 的额外优点对比着 FSDP 说。

shard N 份的 TP 和 FSDP，都节省了模型侧的参数量到 $\frac{1}{N}$，但 TP 因为使用的是切片后的矩阵来做投影，所以产出的中间激活值也变小了，会减小每张卡上激活值的大小。
比如对于一个 32B 的 dense 模型，$N = 8$，那么 无论是 TP 还是 FSDP，都能将模型相关的单卡显存占用切到 $\frac{1}{8} \times 32 * 18 = 72G$，勉强装得下。
但如果不对中间激活值做额外的处理。以Qwen2.5 32B 为例做计算：单个 transformer block 的激活值至少（只计算几个投影层的输出，中间的 norm, 激活函数, residual add 等当做用算子融合融掉了）为 $34ld = 0.166 l \times{10^6}$，（直接使用了 megatron 原始论文的结论），在千量级的 token 下，也有大概 GB 级别的占用，64 层的激活值就会占用 64G 量级的显存，必须使用 gradient checkpoint 技术，从而会拖慢训练效率，而使用 TP8 的话，激活值也会近似除 8，处在勉强放得下的情况，配合一些少数细粒度的重计算，就能达到比较好的训练效率。

另外，我们可以看到，TP N 是用 N 张卡算同一个样本，即用 N 张卡的算力，算一个样本的 forward/backward。
FSDP 则相反，它是数据并行的实质，每张卡上用的还是不一样的样本，并且按照前面通算遮掩的分析，每卡上的样本还不能太短，否则遮掩不住。
我们知道，训练的 batch size 开太大的话，实际上是一种数据浪费，因为稍小一点的 batch size 已经能得到稳定的梯度估计，同样的数据量，batch size 小一些能训得更久。
所以如果你卡量且希望把他们都用起来的情况下， FSDP 会强迫你的 batch size 变大，要和卡量等比例的增大，而 TP 正相反 TP N 下，global batch size 反而是除8了，卡足够的情况下这种 “N 张卡算同一个样本” 的并行方法能帮助你更快跑出结果。

### 通信量分析之外的通信开销

在 megatron 的相关系列论文中，直接比较过 deepspeed zero3：

![megatron3d_vs_deepspeed_zero3](https://ar5iv.labs.arxiv.org/html/2104.04473/assets/x14.png)

虽说朴素的 zero3 就是跨全节点的 FSDP，但前面我们分析过，即使跨节点通信，应该也是出于能跑的状态，且 FSDP 的单卡通信量，跟参与切分卡量无关，卡多的时候，虽然要发送的次数更多，但是单卡上的分片也更小了，一乘一除就会对消。
所以只要计算强度还在，就应该能遮掩住，不应该出现上图这样，随着卡量增加， zero3 场景下，单卡 FLOPS 明显降低的情况。

这里的原因在于说，通信时间，除了通信量/信道带宽之外，还会有一个额外的常数项，是收发两端建立连接的等控制面固有的时延，一次收发实际上的时间应该是 $a + bX$ ，其中 $a$ 就是收发两端握手建立连接的固有时延，$X$ 是实际的通信量。
对于 zero3，卡量翻 $N$ 倍， 对一张卡的每次发送，$X$ 确实会对应除 $N$，而发送量翻 $N$ 倍，所以其实总时延变成了 $N(a + bX/N) = Na + bX$，当 $N$ 特别大的时候 $Na$ 这一项就不能忽略了，之前计算得到的各种 $T_\text{compute}$ 就会对应的增大，从而导致通信计算不能完全遮住，出现 zero3 卡量越多，单卡效率显著降低的问题。

目前使用 FSDP 推荐的配置其实是，在显存允许的情况下，只在节点内 shard 8 份，节点内的通信更快，同时也控制了 FSDP 这个并行维度中 $N$ 的无限增长，导致的控制面时延放大，影响训练效率。

总结来说就是，对于一个固定的通信量，一次性把它通信完和分 $N$ 此每次通信 $1/N$ 在实际运行的效果也是不一样的。
这样抽象之后来看，其实不止 zero3/FSDP 可能存在这种问题， TP 甚至朴素的 DP 也会存在这个问题。

对于 TP， 我们前面提到它的通算遮掩是通过实际执行矩阵乘法的时候，先执行某些块，算出结果，发送进行 all reduce，然后再执行其它块来达成目的。
这其实是一个多次但是小包的通信过程，但只是因为它跑在 nvlink 这种建立连接都很快，$a$ 很小的信道上，才不会明显感知到效率问题。

对于 DP，我们知道即使是朴素的 data parallel，也有在 backward 后同步所有 rank 梯度的 allreduce，虽然 all reduce 的单节点通信量不随 DP size 的增大而增大，但通过前面的分析我们知道， DP size 特别大的时候，每次建立连接的时延就不可忽略了，是有可能拖慢训练的。
一种解决方案就是采用跨节点的 PP，将模型分 PP 后，做梯度的 all reduce 就只用在同样的 PP rank 的子模型里做，减小了 DP size，又可以把链接建立的时延做回到影响比较小的范围。
但 PP 本身的气泡遮掩，又是另一个话题了。

另外这里也可以看到，在 2020年前后 megatron & zero 刚发布的时候，大家做系统相关的文章都在猛地证明自己的框架能 scale 到几百B 甚至 1T 的 dense 模型还能跑出很好的效率，但在2025 年结束，2026 年开始，我们知道其实这里有些背离实践需求了，做那么大的 dense model 收益其实很低，点提不了多少，但 latency 增高用户体验实打实的差了一截（参考 llama3 405B）

### 计算量分析之外的计算开销

这里解决最后一个伏笔，为什么前面刻意提到的是 Qwen2.5 而不是 Qwen3。
我们知道 Qwen3 开始有 MoE 的模型，而目前一个很标准的面试问题是 “为什么 MoE 模型常开 EP 而不开 TP”。
本节的分析内容就能回答这个问题。

在前面的分析过程中，我们都是假设每张卡按着标称的 FLOPS 在满负载运转，但实际上，即使是 shape 优化得最好的 GEMM，一般也就最高能达到 80% 的 MFU，即标称值 80% 的计算效率，因为实际的运算中，要从 memory 读数据，再写回结果。
而通信一般是确实能接近打满标称值的。
但另一方面，我们并不希望我们实际使用的是比较低效率的 GEMM，因为这会拖慢实际的模型训练效率，而我们可以打表计算一下不同 $M,K,N$ 组合下的 GEMM 的效率。
不同卡型结果会不一样，但一般来说，K 比较小的时候，GEMM 的 MFU 会降低。
所以，即使我们前面算出来切分后的 hidden size 的分野值在几百到一千，我们实际也不会把 TP 开很大，使得 hidden size 接近这个值，因为这个 size 的 GEMM 的效率，本身很低，从而拖慢了训练效率。

在 MoE 中， FFN 的 size 选型发生了变化，比如 Qwen3 的 30BA3B，单个专家的 intermediate size 是 768，即使是旗舰的 235BA22B intermediate size 也才 1536，deepseek v3 671B 的 intermediate size 也就 2048。
它们即使不切分，也运行在一个相对低效的 GEMM 区间了，而对 MoE FFN 做 TP，也就是把每个专家都做 col wise & row wise 的切分，随便一切分，本就已经卡在比较低效率区间的 intermediate size 就会更小，单个 GEMM 运行在更不适合的 shape 上，从而更影响整体的训练效率。
与其使用 TP，不如使用 EP，对 token 做路由，来减小每张卡上的显存开销。

## 参考

- https://zhuanlan.zhihu.com/p/1964984297116144859
- https://jax-ml.github.io/scaling-book/training/
- https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html#when-and-why-you-should-apply-tensor-parallel
- https://zhuanlan.zhihu.com/p/1979141713449742500
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
