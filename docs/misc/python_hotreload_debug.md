---
title: 热重载调试 python 程序
keywords: python, debugger, hot reload
---

## 背景动机

在调试一些有长初始化过程的 python 程序的时候，朴素地编写-断点调试-得到反馈的过程，每次都要重启整个程序，非常耗时。
常常会出现如下的循环：

1. 启动调试任务，等待程序运行，先经过长时间的初始化，最终程序运行到断点处。
2. 做一些断点变量的查看，和在 debugger 内尝试新代码，找到修改的方向。
3. 重启整个程序，回到步骤 1，重复。

美好的时光大部分都耗散在了等待程序反复地初始化。
对于最近的我来说，以上循环发生在调试 flux/sd3.5 的推理程序，尝试不同参数和推理策略，查看效果之中。
读取这些模型组的权重，需要花掉一分钟左右的时间，反复的重启、调试体验确实不好。

其实我有一些调试这种程序的经验。
比如可以先不 load 权重，用 `torch.device('meta')` 来初始化空模型，先对齐推理步骤时每一步的 tensor shape，保证能运行通，再 load 权重看效果。
又比如多在代码里写注释，加 type hint，使得每次往前写代码时，都能更好地想象当前上下文运行的情况，让自己一次能自信地写更多内容后，再一次性调试校验。
又或者用 jupyter notebook 先探索尝试，等代码稳定后把代码块提取出来，形成脚本。

这些都是可行的方案，但或多或少有不尽人意的地方。
以前做游戏程序，还有业余学些前端的经历告诉我，这个问题是很适合用“热重载”来解决的。
如果能有一个热重载调试工具，以上的循环就会被打破：

1. 启动调试任务，等待程序运行，先经过长时间的初始化，最终程序运行到断点处。
2. 做一些断点变量的查看，和在 debugger 内尝试新代码，找到修改的方向。
3. 修改代码，等待热重载，回到步骤 2，重复，因为跳过了长时间的初始化，新的代码很快生效，就能很快地找到修改的方向。

所以问题就变成了怎么在 python 里搞热重载调试。

## 实现

直接搜索 python hot reload，就可以发现一个好用的库 [`jurigged`](https://github.com/breuleux/jurigged)。
使用它来启动调试和使用 `pdb` 基本是类似的体验。

简单地在命令行打 `jurigged <xxx.py>` 启动调试。调试过程中你对脚本的修改，会立即生效，正如它 github README的动图所展示的那样，简明直观。

美中不足的是它不是 pdb，也不是 vscode 官方插件的 debugger debugpy，也没和它们有很好的联动（但我觉得稍加改造并不难将其提升）。
这些我用了很久的 debugger 还是有很多其他我难以割舍的功能。
在尝试搜索了发现了 vscode 的 debugger 是有这样的功能的，但是被官方称为 `autoReload`

使用的方式很简单，在 launch.json 你需要热重载的 debug 任务的字典中中添加一行`"autoReload": {"enable": true}` 形如：

```
{
    "name": xxx
    ...
    "autoReload": {"enable": true}  
    ...
}
```

就可以启动热重载调试。

你修改的每一个函数、成员方法... 都会在文件保存的时候，触发热重载，替换掉运行中程序地相应的函数对象、成员方法对象，在你**下一次调用**的时候，就会调用的是新修改的函数了。
同时，这也暗含着你可能需要对你的脚本做一定的修改，才能最大程度利用好这个功能，以下是一些提示：

1. 对于 web 类程序，因为 web 框架通常自带了事件循环和异常处理，基本不需要代码侧的额外改造。
2. 对于脚本类程序合理地把代码封装函数，裸脚本一行行写吃不到太多的热重载动态替换函数的好处。（同时也有助于脚本可读性提升吧）
3. 对于脚本类程序用一个死循环 + try except 来先包裹除了耗时代码之外的其他代码，防止自己改了可能会触发异常的代码，带崩整个调试，导致需要重启调试，等待耗时地初始化。

比如一个 sd3.5 的改参数跑图的脚本，可以这样写：

```python
from custom_sd3 import CustomSD3Pipeline  ## 自己的推理 pipeline 实现
## from diffusers import StableDiffusion3Pipeline ## 魔改自官方的 pipeline
from itertools import cycle

def main():
    args = parse_args()
    pipeline = CustomSD3Pipeline.from_pretrained("xxx")
    dataset = init_dataset()
    for data in cycle(dataset):
        try:
            image = pipeline(  # vscode 的断点打在这里，
                xx1=args.xx1,  # 这里也尽可能地从变量来设置参数，而不写死 xx=某个常量，
                xx2=args.xx2,  # 这样在 debugger 中就可以修改变量，进而修改入参，并立即查看效果
                ...
            )
            save_image(image)
        except Exception as e:
            import traceback; traceback.print_exc()
            continue

if __name__ == '__main__':
    main()

```

去修改 `CustomSD3Pipeline` 的 `__call__` 函数的时候，就可以随性地修改，改到一半大脑过载想不明白了，就先保存，再按一次 `f5` 回到死循环体里的下一次函数调用，`f10`，`f11` 步进到当前位置看看实际的运行情况是怎样的，再想下一步修改。

## 原理小探

vscode 官方 python 插件叫做 [`debugpy`](https://github.com/microsoft/debugpy)。
通过阅读它的源码的相关章节就可以窥探它的一些原理实现。

1. 由一个 [`Wathcher`](https://github.com/microsoft/debugpy/blob/main/src/debugpy/_vendored/pydevd/_pydev_bundle/fsnotify/__init__.py) 在它的线程里查看项目目录中的源文件是否被修改，若修改触发热重载替换有更新内容的代码的回调函数，即 2 中的相关内容。
2. 在 [`pydevd_reload.py`](https://github.com/microsoft/debugpy/blob/main/src/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_reload.py) 文件中实现了重新 import 文件，查找差异函数，并替换到当前运行的程序的功能。

再往下就可以展开更多的细节，在此列举一二：

1. 如何判定文件进行了修改：利用操作系统的接口，查看了文件的修改时间(mtime)如果说和上次记录的mtime不同，就认为文件被修改了。
2. 如何判定函数被修改：将修改了的文件在一个额外定义的空的伪造的 global dict 里做执行，执行的结果就是这个模块更新后的结果，从中可以得到函数对象，通过比较同名的函数对象的 code object 相关属性来判定函数是否被修改。
3. 如何替换函数：若函数发生了修改，则将上面的新的函数对象的 `__code__` 或者 `func_code` 赋值给老函数的 `__code__` 或者 `func_code`。


## 总结

python 中有很多支持热重载的 debugger，本文重点介绍了 vscode 官方 python 插件的 debugger debugpy 的热重载使用方式，希望它能给你带来更好的 python 调试体验。
