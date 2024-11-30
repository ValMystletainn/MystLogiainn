---
title: jsonargparse 库介绍
subtitle: 实现自己的 trainer 时的新发现
keywords: python, deep learning, programming
---

开宗明义，先把要安利的库放上来：[jsonargparse](https://jsonargparse.readthedocs.io/en/stable/#) 

## 背景

Trainer 是包裹常见的深度学习训练流程的一个类，它的目标是通过提供高抽象的训练流程接口，**减少用户所需编写的模板训练代码**，只需要针对性的修改 model/dataloader/loss 等部分，即可实现快速训练。


知名的模型如 huggingface transformers, detectron2, mmcv 都提供了类似的 Trainer 类，还有 pytorch-lightning 这种以写 traniner 起家的模型训练框架库。
但这些库的潜在问题是过度封装，尤其是当你需要对训练过程做大规模修改的时候，这个问题会被放大很多。比如：早期用 pytorch-lightning 训练 gan，就得把 generator 和 discriminator 都放到一起封装成一个 `pl.LightningModule` 类，并且修改一下所用的 optimizer，让它每次只更新一部分模型参数，对应于原始算法的更新 G 和 更新 D 的部分，才能做到合理的 GAN 训练。

因此，很多人的选择是自己写一个 trainer，把细节把握在自己手中。如这两篇知乎文章 [300行代码实现一个优雅的PyTorch Trainer - serendipity](https://zhuanlan.zhihu.com/p/449181811)，[Pytorch封装一个优雅的Trainer类，并实现可扩展性 - 新生代农民工的文章](https://zhuanlan.zhihu.com/p/414843341)。
而我自己也实现过一个 trainer [torchGBA](https://github.com/ValMystletainn/torchGBA)，并在读书时在实验室的小范围内，作为深度学习项目管理的新人入门案例分享过。

在最近，我想重构这个 trainer，把现在看来写得不成熟的地方做些修改，在方案调研时，发现了 pytorch-lightning 用到了 jsonargparse 这个库，十分符合我的需求，就有了这篇文章。

## 痒点

写 trainer 是为了用一套代码，灵活切换多个相似的模型数据集、训练策略等因素。
使用一个或多个配置文件，来做模型、数据等对象的管理很方便事后的对比和复现。
我之前的 trianer 实现里大概是这样做的。

```python
parser.add_argument('--model_config', type=str, default='configs/model.yaml')
parser.add_argument('--train_data_config', type=str, default='configs/train_data.yaml')
...  # parse, 打开相应的配置文件，解析成 dict
model = getattr(model_module, model_config['class_name'])(**model_config['init_args'])
```

但在实际训模型的时候，往往会出现这样的情况：有几个参数我要对比一下，刚好机器资源很丰富，所以我开了多个训练容器，试图同时训练这些配置。
这样的代码就需要我多次打开、修改、关闭配置文件，在不同的训练容器里训练。

本次重构的重点就是这个小功能：如果传入 `--model.xx yy` 这样的参数，就对应修改加载后的配置字典的相应值，使得我不用打开配置文件，也能修改一些训练参数。

这个需求已经被很多配置库解决了，之前我试图用 `omegaconf` 实现过，参考的是 [omegaconf-merge](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#omegaconf-merge) 的部分。
使用 `OmegaConf.from_cli` 可以读出`xx.yy=zz` 这样格式的命令行参数，然后用 `OmegaConf.merge` 覆写从配置文件里读出来配置字典里，就能完成这样的要求。

`pytorch-lightning` 的 `lightning-cli` 也有相似的功能。
但它的用户体验更好，[入门文档](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html#) 。
cli 里会有很多 `--help` 能打出来的提示，在大到 cli 入口可选 `fit`, `validate`, `test`, `predict`, 再细节到选定一个功能后，能传入什么模型，什么数据集，再小到模型选择具体的一个类后，有什么参数可供修改，都能在相应等级的参数后接 `--help` 查到。
长期维护和多人协作时，这样的精细的 `--help` 很受用，免去了自己遗忘或合作者不知道模型/数据集的有什么参数可以设置，需要再翻源码的麻烦。

在查看了 `pytorch-lightning` 的源码后，我发现这个功能基本都是靠 `jsonargparse` 这个库来实现的。
详细阅读了它的文档后，我就来尝试为自己的 trainer 补充这样的功能了。

## 使用介绍

`jsonargparse` 有两个核心接口， `ArgumentParser` 和 `CLI`。

`ArgumentParser` 继承并强化了标注库 `argparser` 的 `ArgumentParser`，增加了对自定义类作为 argument 的`type`参数 的合理支持，增加了把命令行参数导出为 .yaml 配置和读入 `.yaml` 等功能。
（对，虽然库叫 jsonargparse，但默认导出的配置格式是 yaml 而不是 json）

另一个接口 `CLI` 通过读入一个或一些函数或类，通过函数的入参的名字和类型，自动生成 `ArgumentParser` 实例，然后把命令行参数解析成函数入参的格式，最后调用相应的函数。
实际使用中我觉得 `CLI` 这个入口是最方便优雅的，很符合文档里所写的 **Non-intrusive/decoupled, Minimal boilerplate** 两个设计原则。

吹了那么多，直接来看代码案例吧

```python
## import ...

def train(
    model: nn.Module,
    ...
):
    """
    train a model

    Args:
        model: model to train
    ...
    """
    pass

if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(train)

```

在命令行里，`python main.py --help` 就能看到 `train` 函数的入参的名字和类型，以及默认参数的默认值，还有如果函数有按照PEP 257 的规范写 docstring 时，每个参数的在 docstring 里的对应描述。

上面的示例代码运行 `python <脚本名> --help` 后的输出格式是类似这样的

```
usage: <脚本名> [-h] [--config CONFIG] [--print_config[=flags]] [--model.help CLASS_PATH_OR_NAME] model

train a model

positional arguments:
  model                 model to train (required, type: <class 'Module'>, known subclasses: ...
```

而通过脚本的 `--print_config` 这个选项，就能把相应的参数和模型配置打印出来，重定向到一个 `.yaml` 文件里就得到了配置文件的修改起点。
通过`--config <参数名>` 就可以选择相应的配置文件，当然选择完以后还可以在后面紧跟 `--参数名.子参数名  参数值` 来动态覆写部分参数。

比如当我传入以下 `yaml` 文件到上的示例脚本，在主函数里打印一下 `model`，就能得到 `Linear(in_features=10, out_features=10, bias=True)` 发现确实选择了合适的模型类，传入了正确的参数，并做了实例化。
而在命令行里做参数覆写，新增 `--model.out_features 20` 就可以把 `out_features` 这个参数从 10 改成 20 并做相应的实例化。

```yaml
model:
  class_path: torch.nn.Linear
  init_args:
    in_features: 10
    out_features: 10
```

## 意外收获

这个库也意外解决了之前的另一个写脚本时有不满的地方，就是 `argparse` 解析出来的 `args` 是没有类型标注的，每次写 `args.xx` 都没有 ide 提示，都很不方便。

`jsonargparse` 通过把一个函数的入参和签名做出命令行参数，就能让用户在写程序的时候读取的面对的是函数入参，很直接地就能得到写 `typehint` 的好处。
类似处理的库还有 `fire`, `Typer` 等，也是通过解析函数的签名，构造 `ArgumentParser` 来做接口实现。

## 源码解析

这里以传入单个函数给 `CLI` 的情况为示例，感兴趣的读者可以边阅读文章，边自己打断点试着运行一下，相互校验。

当 `CLI` 传入一个函数，并被调用时，主要的流程是

1. 根据`CLI`的默认 `parser_class` 初始化了一个 `jsonargparse.ArgumentParser` 实例 `parser`。
2. 为 `parser` 添加一个 `--config` 的选项，用来读取配置文件。
3. 通过 `_add_component_to_parser` 函数，使用传入的函数和 `parser` 实例，为 `parser` 增加新的参数组。逐层看下去，里面调用的是 `parser.add_function_arguments` -> `parser._add_signature_arguments` 方法，最后落到用 `inspect` 库的相关功能，解析一个函数的签名，生成相应的 `ArgumentParser` 实例。
4. 调用 `parser` 的 `parse_args` 方法，解析命令行参数，把参数放到相应的参数组上。
5. 调用 `parser` 的 `instantiate_classes` 方法，根据 `parser` 上的参数，实例化相应的类，也是把参数递归地转换并构造字典，然后用 `**init_dict` 的形式来做构造。

## 进阶使用

事情到这里，就结束了吗？我能用这个库很好地重构我的 trainer 吗？
之前展示的都是很浅的例子，在实际的模型训练中是不够用的。
举个例子：

怎么实例化 optimizer，dataloader？
`CLI` 是在自己的函数体内部先把入口函数所有入参都实例化了，再调用入口函数。而 optimizer 和 dataloader 的特点是，一些参数（如学习率，batchsize）是可以在配置文件中事先给定的，但有另一些参数是需要其他对象实例化后才能拿到的，如 optimizer 的 `params` 需要 model 实例化后调用 model.parameters(), dataloader 的 `dataset` 需要实例化后的 dataset。

抽象来看，这都要求我们有，“固定一部分参数，另一些参数留待传入，最后再执行调用” 的能力，类似于 `functools.partial` 的效果。
幸运的事是 `jsonargparse` 还真有这个功能。
在文档中某个[不起眼的小节](https://jsonargparse.readthedocs.io/en/stable/#)有记载。
简单来说，我们将类型标注符写成 `Callable`，就能触发。
（偏函数构造从实现的角度看，构造偏函数的函数，等价于“一个函数，它的行为是，吃带绑定部分参数的函数和绑定的部分参数，返回一个参数自由度少了一些的函数”，返回类型还是函数，自然可调用，标注为 `Callable`）

那么实际上，就可以这样写训练函数的签名和做实现

```python
def train(
    model: nn.Module
    optimizer: Callable[[Iterable], Optimizer]
    ...
):
    ...
    optimizer = optimizer(model.parameters())
    ...
```

而相关的配置文件类似可以写出这样
```yaml
model:
  class_path: torch.nn.Linear
  init_args:
    in_features: 10
    out_features: 10

optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: 0.001
```

跳出来看，它提供的就是一个延迟实例化的功能。让参数解析往回走了一点，只做入参的解析和结构化，帮我们把类导入好，把参数放好，至于什么时候做实例化，是用户自己决定的。

在我接触到的实际训练场景中，这个操作是很有用的。
比如说，当我要续训一个模型，模型又很大。往往初始化的步骤是这么写的

```python
with torch.device('meta'):
    model = Model(**init_args)
model.load_state_dict(torch.load('checkpoint.pth'), assign=True, mmap=True, map_location=device)
```

在`meta device` 上实例化模型只会初始化参数的大小，没有实际的参数值占用，不用执行耗时而无用马上会被覆盖掉的参数初始化操作。
后面 load_state_dict 操作，会把参数值从硬盘里加载到 `device` 上，并且 `assign=True`是让模型直接使用这一片内存作为参数的实际值，没有额外的拷贝开销。
在这个场景下，我确实需要模型绑定好模型层数，大小相关的超参，但又不希望它在进入入口函数的时候已经被实例化了。
那训练入口函数就应该改成

```python
def train(
    model: Callable[[], nn.Module],
    ckpt_path: Optional[None]
   ...
):  
    ...
    if ckpt_path is not None:
        with torch.device('meta'):
            model = model()
        model.load_state_dict(torch.load('checkpoint.pth'), assign=True, mmap=True, map_location=device)
    else:
        model = model()
   ...
```   

还有部分超参数可能是在多个实例中共享的，可以参考文档里的 `Argument Link` 或者 `Variable interpolation` 来实现。

## 无招胜有招

回到最开始的问题，我要重构我的 trainer 吗？

前文引用的 trainer 实现知乎文章中的评论区中有提到的

> torch就是因为支持functional的写法才会让它被广泛使用吧，又封装成trainer不就又改到keras-tf2那个样式了么。。。。而且你有没有听说过一个东西叫做Pytorch-lighting。。。。

当时我的理解还是 `functional` 的写法就是指用脚本或单个函数来写训练过程，写损失函数，一些简单算子。

后来对各种在实践中对各种编程范式理解更深了以后，发现写成 oop 类型的 trainer，self 里就有全局的状态，使用 `self.some_func` 做一些局部的操作的时候，很难确定它是不是通过 self 改变了很多非其他入参的属性，做了全局的状态迁移。
某个成员变量发生没发生更改，甚至有没有这个成员变量，都是未可知的。

deep learning 的训练的很多子模块大抵还是一个很接近于 fp 所描述纯函数的样子。
输入数据传入模型，就应该返回输出，输出结果和 ground truth 给到损失函数，就应该输出具体的函数值，只要入参是一样的，出来的结果理应是一样的。
(当然实际也不是真的纯函数，因为 pytorch 会在后台构造计算图，是有程序状态的变化的，但对常规使用来说无感)

用 jsonargparse 的接口，更能把训练过程包装成一个函数，明明白白地提醒自己，这里有什么变量，我应该用它来怎么操作。
想要多打一些 log，就多写几行代码，如果实在很多了，就包装成一个函数列表，在配置文件里指定，就好了。

项目构造就会简单的变成类似 huggingface 的几个库的样子，结构清晰，方便复制。

```
.
├── configs/...
├── src/项目名  # 用做类库
│   └── models
|       |── __init__.py
│       ├── model1.py
│       └── model2.py
|   └── datasets/...
└── scripts  # 用做脚本
    └── train.py

```


## 总结

本文介绍了一个简单而优雅的库 `jsonargparse`，它能让我们把训练过程的入参，和训练过程的逻辑，分离到配置文件里，方便地实现各种训练的复现，和配置对比。
同时也打消了我重构 trainer 的念头，用 `jsonargparse` 来规范化的训练脚本，本身就是很棒的 trainer 设计！
