---
title: 提升阅读体验的 markdown 黑魔法 MyST
subtitle: 更新更好的 markdown 体验
short_title: MyST 介绍
keywords: myst, markdown, blog
---

(sec-background-movtivation)=
## 背景动机

最近想更新一下自己的博客
我原来用的方案是 [obsidian-zola](https://github.com/ppeetteerrs/obsidian-zola)，这是一个 markdown 写内容的静态站点生成器。
使用它的体验还是很好的，它有完整的 common mark 支持，也有 latex 公式的渲染，而且也和 obsidian 的双链图有很好的联动。
丰富的引用链接确实会让博客文章的整体性更强，但当引用链接多了，就会带来很多引用链接的点击和跳转。
来回地跳转潜在地打断了本该连贯的阅读，这是值得改善的。（这事在读论文的时候也很困扰我）

一个自然的解决方案是，在跳转链接上做一个悬浮预览窗，减少来回跳转的次数。
由此，我搜到了一个相关的知名前端库 [tippyjs](https://github.com/atomiks/tippyjs)。
但它只是提供了一个悬浮显示的插件标签，悬浮显示的内容还需要我们自己填写。
所以想在原始的博客方案中使用这个库，就得去魔改 obsidian-zola 的解析渲染部分的逻辑，截获相应的内容，并插入适当的 tippy 控件。
这有一定工程量，那不如先看看使用 tippyjs 的其他仓库，看看有没有人已经做了类似的事。

这一次的检索带我来到了 [sphinx-tippy](https://github.com/sphinx-extensions2/sphinx-tippy) 这个仓库。
我详细了解了一下 sphinx 生态，顺便给这个 sphinx 插件修了个 bug。
Sphinx 下很知名的一个主题是 [jupyter-book](https://github.com/jupyter-book/jupyter-book)，并由此接触了这个社群，发现了他们正在活跃开发的 jupyterbook 2.0 项目 [mystmd](https://github.com/jupyter-book/mystmd)。
在我看来，它相较于其他的 markdown 静态站点生成器有以下优势：

1. 引用预览。悬浮在引用上，就可以预览引用内容。并且它做得更极致，可以在预览页面递归地往下查看“预览页面里的引用”的预览。
2. 更多学术特化。它增加了规范化的图注语法；规范的公式表情标注；对引用链接是 doi，wiki 等内容也做了预览解析...
3. 支持运行代码内容。用远程的 jupyter hub/binder 或者 jupyterlite 来支持内容的运行，可做交互性更强的内容。
4. 可导出成 latex typst 来转 pdf, 而不止支持导出为站点，方便做传统场景的内容草稿导出。

当然 mystmd 也有一些不足，但感觉能靠 pr 解决：

1. 评论区，博客还是有评论感觉才像博客。静态博客常用的 giscus 和 utterance 暂时还没得到支持。（但 jupyterbook 是有这些插件的）
2. 代码的 in browser 运行。 jupyterlite 是靠 wasm 来运行 python 和 python 的 c extension 的。没有额外编译支持的包就不能 in browser 运行了。但个人常用的 `numpy`, `matplotlib` 等科学运算包都是有 wasm 编译版本的，只有 `pytorch` 还未得到支持。

(sec-resources)=
## 上手资源

- 官方文档：[https://mystmd.org/](https://mystmd.org/)。 它能解决 80% 到 90% 的问题
- 示例项目：[tlke-finitevolume](https://simpeg.xyz/tle-finitevolume/), [myst introdcution in scipy proceeding 2024](https://proceedings.scipy.org/articles/018fcf90-7d9b-73ac-8b31-ecb257c2c98f#fig-export), [2i2c year reports](https://2i2c.org/report-czi-2021/year2#artifacts-publications-and-software-code) 等（但感觉博客类网站偏少）

(sec-basic-usage)=
## 基本使用

本节用于展示 mystmd 的一些使用语法案例和效果。
由于 mystmd 的扩展语法比较多，而且达成同一个效果的方式多种多样。
这里展示一些我常用的语法子集。
子集的选取原则是，尽可能做到与 [CommonMark](https://en.wikipedia.org/wiki/Markdown#Standardization) 和 [GFM](https://en.wikipedia.org/wiki/Markdown#GitHub_Flavored_Markdown) 的语法保持兼容，若不一致则渲染后至少能看出原始语义。

(sec-pure-typing)=
### 纯文字书写

按照如下约定写。

1. 分段：用换行加一个空行来分段，用换行不加空行来分割句子。分割句子主要是为了语义清晰和 git 追踪友好，并不必须。
2. 标题和分节: 还是使用 一个或多个 `#` 来起不同等级的标题。但由于博客还需要抽取每篇文章的标题放在侧边栏导航，所以尽量只用一个一级标题来作为全文标题，如果在 frontmatter 里手工设置了全文标题，则不使用一级标题。
3. 斜体、加粗、内联代码。常规的使用 一对 `*`，两对 `*` 和一对 \` 来包裹相应的内容。
4. 脚注。通过`[^脚注标签]`在正文文本中插入脚注编号，用 `[^脚注标签]: 内容` 来写脚注内容。

(sec-figure)=
### 图片

mystmd 支持常规的 markdown 图片插入语法，也即 `![alt text](url)`。
在更学生的场景里，人们通常会在图下方加入图注标题，并给图打上标签编号，方便在后文中引用。
在这里，我会使用 mystmd 扩展的指令语法 (directive) 中图片相关的语法。通过如下格式来实现图注和标签编号。

```
:::{figure}
:label: 可选，图片的标签，用于引用时指代，一般用 fig-xx 的格式写
:width: 可选，填 xx% ，用于指定图片的宽度
![alt text](url)

图片的图注标题
:::
```

用这样的格式，就能实现如 @fig-fruit 所示的带图注，支持用 `@标签名` 或者 `[](#标签名)` 来引用的图，并最大程度在不支持 mystmd 的平台上，也能正确渲染图片部分。（因为由冒号展开的指令块在这些平台上会被当成普通的文段，从而解析出原来的 `![alt text](url)` 语法）

:::{figure}
:label: fig-fruit
![Here is some fruit 🍏](https://github.com/rowanc1/pics/blob/main/apples-wide.png?raw=true)

水果！
:::

此外，我自己的仓库里还实现了一个插件，通过调用 [kroki](https://kroki.io/) 来实现各种描述语言到图形的转换， @fig-kroki 就是一个例子。
一般我常用的是 tikz 和 excalidraw 两种格式。
前者是对公式友好的绘图格式。
后者是因为可以在它提供的网页端电子白板或 vscode 插件，用于所见即所得的绘图和导出绘图后的元素排布文件，作图较为方便。
我使用 kroki 转换的所选的图像格式都是 svg，理论上来说 svg 的图是能选中图中的。
但由于 mystmd 的插入图片方式实现的问题，在博客原页面中是无法做到的，但可以通过右键图片在新标签页中打开实现。


````{figure}
:label: fig-kroki
```{kroki}
:src: blockdiag
:alt: "blockdiag"
:align: center
:width: 60%

blockdiag {
  Kroki -> generates -> "Block diagrams";
  Kroki -> is -> "very easy!";

  Kroki [color = "greenyellow"];
  "Block diagrams" [color = "pink"];
  "very easy!" [color = "orange"];
}

```

[kroki](https://kroki.io/) 给出的 BlockDiag 渲染后的图
````

(sec-math)=
### 数学公式

CommonMark 是没有公式支持的，但是 GFM 和大部分扩展的 markdown 解析器都支持公式语法。
通常来说最小的支持就是用 `$` 来包裹内联公式和用 `$$` 来包裹跨行公式。
在 MyST 里既支持这些语法，也有一个 `math` 的指令块，类似于 latex 的 `\begin{equation}` 和 `\end{equation}` 来包裹公式和渲染和控制公式的编号。
但 `math` 指令块和 `figure` 指令块的设计不同，里面不能再用 `$$` 来包裹公式，只能直接做公式编码。
处于兼容性考虑，我并不使用这个指令，而是使用 `$$` 来包裹公式，然后用前置的 `(标签名)=` 来给公式块加标签，即：

```
(标签名，通常是eq:xxx)=
$$
公式内容
$$
```

使用这个语法，可以得到式 @eq-maxwell 所示的麦克斯韦方程。

(eq-maxwell)=
$$
\begin{aligned}
    \nabla \cdot \mathbf{E} &= \frac{\rho}{\epsilon_0} \\
    \nabla \cdot \mathbf{B} &= 0 \\
    \nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
    \nabla \times \mathbf{B} &= \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}
\end{aligned}
$$

此外，myst 还提供定义、定理、证明、公理等多种环境块，跟 LaTeX 中相关功能很类似，具体可以参考官方文档 [proofs-and-theorms 小节](https://mystmd.org/guide/proofs-and-theorems)。

(sec-cross-ref)=
### 交叉引用

使用特定语法块指令块中的 `label` 字段，或者在任意内容块之前使用 `(标签名)=` 就可以为内容块定义标签。
然后使用如下两种语法都可以引用标签。

1. `@标签名`: 该方法比较简单直觉，优先使用
2. `[](#标签名)`: 该方法比较冗长，但可以引用不同文章里的标签，即用 `[](其他文章的无后缀路径名#标签名)` 来引用

引用外部链接时，和 Commmark 一样可以使用 `[showing text](url)` 来引用。
但 MyST 对学术向的引用做了一些特化。
详细的指引可见[官方文档中的相关章节](https://mystmd.org/guide/external-references)，这里我摘录我常用的功能：

1. 当引用的链接是 wikipedia, doi, github issue/pull request时，会自动解析并给出悬浮预览，此外不写 showing text 也会填写自动解析的内容（wiki的标题，doi 的作者和发表年份）。但处于兼容其他 markdown 语法的考虑，不建议不写 showing text。
2. 引用 doi 后，会自动在文章结尾生成一个参考文献列表。

如： [ResNet](https://doi.org/10.48550/arXiv.1512.03385) 是何凯明的工作，它被广泛应用于当今的深度学习模型中，这是它的[维基百科词条](https://en.wikipedia.org/wiki/Residual_neural_network)，最原始的实现还是使用的 caffe，比如这是原始代码库里的 [ResNet50的caffe声明](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt)

(sec-code-block)=
### 代码块

mystmd 支持常规的 markdown 代码块语法，即用一对 \` 来包裹行内代码，用一对 \`\`\` 来包裹代码块。
用一对 \`\`\` 来包裹代码块可以在起始行加上语言名，如 `python`，`bash`，`cpp` 等，来在渲染时得到代码高亮。

此外，mystmd 还有三个代码相关的指令块，`code`, `literalinclude` 和 `code-cell`。
它们分别用于

1. `code`：常规 markdown 代码块的扩展版本，用于额外添加代码块标题（caption）和用于引用的标签（label）和行号高亮等属性。
2. `literalinclude`：从文件中读取代码，并渲染成代码块。同样支持配置标题，标签和选择部分引入的行号和高亮
3. `code-cell`：连接远端 jupyter kernel 由 jupyterlite 在网页端运行的代码块。

处于兼容性考虑，我一般不使用指令`code`，而是使用 `\` 来包裹代码块，并使用 `(标签名)=` 来给代码块加标签。
但`literalinclude` 和 `code-cell` 还是会使用的。
它们的使用可见官方文档：[code 和 literalinclude 的文档](https://mystmd.org/guide/code-blocks), [code-cell 的文档](https://mystmd.org/guide/notebooks-with-markdown)


### admonition & callout

admonition 和 callout 是从 JupyterBook 时代在 sphinx 就引入的一个指令块。
它是一个高亮显示标题和部分内容的块，如用来高亮 tip，warning 等内容。

比如使用如下代码段，可以插入一个 tip 块
```
> [!tip]
> This is a tip!
```

得到这样的效果

> [!tip]
> This is a tip!

我只选用了和 github 的 GFM 兼容的语法，使得文件在 github 上浏览时也能最大程度地正确渲染。
但在[官方文档的相关章节里](https://mystmd.org/guide/admonitions)，我们可以找到指令块格式的 admonition/callout 语法，它有更多更丰富的 callout 种类和以及标题、下拉框等更丰富的参数配置，如果有使用这些特性的必要，我就会使用指令块形式的语法。


(sec-summary)=
## 文章小结

这篇文章介绍了 mystmd 中我经常用到的语法功能，大致涵盖了纯文本书写、图片、公式、代码、引用等几方面内容。
它是一个 markdown 之上，借鉴了 rst, latex, tpyst 等多家长处的用于写博客的 markdown 扩展语言，满足了我的很多写博客的刚需。


