# Welcome to wenxu's Blog

访问者你好，这里是我的个人博客。
我叫吴文绪，是一名算法工程师。
这个博客的别名 `MystLogianinn` 是我之前打游戏时 ID ValMystletainn 和 Blog 的混合。

本博客主要用于记录一些日常编程、训模型、读论文的心得体会。
部分内容也会发成[我的知乎文章](https://www.zhihu.com/people/xiang-shuo-sao-hua-de-ren/posts)。
本博客之后可能会扩展别的内容。
比如可以有的内容包括：

- 我的老博客中文章的迁移。
- 做饭心得与菜谱。
- 锻炼打球的思考 ...

本博客基于 [mystmd](https://mystmd.org/) 技术方案构建。
它是一个基于 Markdown 扩展语法的为科技类文档设置的静态网站生成器。
在本博客中，你可以体会到诸如公式、图片悬浮预览，悬浮预览窗套娃，代码块高亮等提升技术博客阅读体验的小功能。

未来，这个博客方案还会增加更多的功能点：

- [ ] 评论 （初步想法是基于 github disscussion/issue 和相关插件，如 [giscus](https://github.com/giscus/giscus)）
- [ ] 基于 [pyodide](https://github.com/pyodide/pyodide) 的纯前端 python 运行，方便交互式地展示文中涉及的代码。其实 mystmd 已经有了对 pyodide based 的各种 kernel 的早期支持，我甚至还探索并写过[相关文档](https://github.com/jupyter-book/mystmd/pull/1507)。但似乎这应该等到它们的功能稳定，以及我自己探索完[无 pthread 支持的 pytorch 编译](https://github.com/pyodide/pyodide/issues/1625#issuecomment-2367597597)，以及对 pytorch 包做一定的压缩，来让 pyodide 可以原形运行 pytorch 后。
- [x] 基于 [kroki](https://github.com/yuzutech/kroki) 的纯文本化绘图

