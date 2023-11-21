# 精度工具

## 目录

- [介绍](#介绍)
- [工具安装](#工具安装)
- [工具使用](#工具使用)

## 介绍

MindStudio精度工具针对模型训练精度问题设计推出了一系列精度工具，包括模型精度预检工具（预检工具）、溢出检测工具、精度比对工具、通信精度检测功能。这些工具有各自侧重的场景，用于辅助用户定位模型精度问题。

### 精度工具各子功能介绍

| 工具名称     | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| [精度比对工具](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools/ptdbg_ascend) | 用于进行PyTorch整网API粒度的数据dump、精度比对和溢出检测，从而定位PyTorch训练场景下的精度问题。 |
| [精度预检工具](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools/api_accuracy_checker) | 用于扫描用户训练模型中所有API，给出各API精度情况的诊断和分析。 |

## 工具安装

[精度比对工具安装指南](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools/ptdbg_ascend#%E5%BF%AB%E9%80%9F%E5%AE%89%E8%A3%85)

**精度预检工具安装指南**：

安装预检工具

将att仓代码下载到本地，并配置环境变量。假设下载后att仓路径为 $ATT_HOME，环境变量应配置为：

```
export PYTHONPATH=$PYTHONPATH:$ATT_HOME/debug/accuracy_tools/
```

环境和依赖

```
pip3 install tqdm prettytable yaml
```

## 工具使用

### 场景一：精度存在偏差

该场景的现象表现为：NPU上训练的网络存在精度问题，精度指标（loss或者具体的评价指标）与标杆相差较多。

对于该场景的问题，可以使用**精度预检工具**或者**精度比对工具**进行定位。

精度预检工具：该工具会对全网每一个API根据其实际训练中的shape、dtype和数值范围生成随机的输入，对比它与标杆的输出差异，并指出输出差异过大不符合精度标准的API。该工具检查单API精度问题准确率超过80%，对比一般dump比对方法减少落盘数据量99%以上。

精度比对工具：在预检工具的输入随机生成的情况下，有些场景预检工具有概率检测不到算子精度问题，这时候除了多跑几次预检工具之外，用户还可以使用dump 比对工具来检测精度问题。具体来说，dump统计量、分段dump、模块化dump，通讯算子dump等功能可以用较轻的数据量实现不同侧重的精度比对，从而定位精度问题。

[精度比对工具使用指南](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools/ptdbg_ascend#%E5%BF%AB%E9%80%9F%E5%AE%89%E8%A3%85)

[精度预检工具使用指南](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools/api_accuracy_checker#%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8F)

### 场景二：训练存在溢出

该场景的现象表现为：训练网络可能存在溢出现象，例如某个step的loss突然变成inf nan，或者混精场景下loss_scale不断减小。

使用方法如下：

1、 使能ptdbg的溢出检测能力。import工具并加上以下代码：

debugger = PrecisionDebugger(dump_path="./dump_overflow_path", hook_name="overflow_check")

关键参数如下：

dump_path：配置数据落盘的路径，必须提前创建

hook_name：必须设置为"overflow_check"

2、 加入启动工具开关代码

溢出检测功能也需要用户模型初始化之后或在训练迭代中配置工具开关：

启动开关：PrecisionDebugger.start() 或 debugger.start()

关闭开关：PrecisionDebugger.stop() 或 debugger.stop() 

3、 运行训练脚本，检测溢出。

工具默认会在检测到1次溢出之后退出训练，将数据保存到相应目录下。

用户可以配置debugger.configure_hook(overflow_nums=-1)来告诉工具在检测到溢出后不退出，这样会检测无限次溢出。在这种场景下也可以在debugger构造时传入step和enable_dataloader等参数来把不同迭代发生的溢出数据存到不同文件夹中。

4、 溢出检测生成的数据

溢出数据同样包括了一个pkl文件和一个保存npy数据的文件夹。其中Overflow_info开头这个pkl保存的是溢出api的简略信息（包括最大最小值等，与dump数据时的pkl格式一致），文件夹中保存的是溢出api的npy数据。