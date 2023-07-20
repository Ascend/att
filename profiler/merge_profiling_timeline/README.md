# Profiling merge tool

## 介绍
本工具支持合并profiling的timeline数据，支持合并指定rank的timline、合并指定timeline中的item


## 1 多timeline融合(常规)

### 1.1 数据采集
使用msporf采集数据，将采集到的所有节点的profiling数据拷贝到当前机器同一目录下，以下假设数据在/home/test/cann_profiling下

profiling数据目录结构示意, 合并timeline必需数据：`msprof.json`和`info.json.*`：
```
|- cann_profiling
    |- PROF_***
        |- timeline
            |- msprof.json
        |- device_*
            |- info.json.*
        ...
    |- PROF_***
    ...
```

### 1.2 合并timeline

可选参数：
- -d: **必选参数**，profiling数据文件或文件夹路径
- -t: **当需要融合多级多卡timeline时需要校准多机间的时间**，传入时间校准的time_difference.json文件路径， 该文件的获取参考[节点间时间差获取](https://gitee.com/aerfaliang/merge_profiling_timeline/tree/master/get_nodes_timediff)
- -o: 可选参数，指定合并后的timeline文件输出的路径，默认为'-d'输入的路径
- --rank：可选参数，指定需要合并timeline的卡号，默认全部合并
- --items：可选参数，指定需要合并的profiling数据项，默认全部合并




**使用示例**：

1、合并单机多卡timeline，默认合并所有卡、所有数据项：
```
python3 main.py -d path/to/cann_profiling/
```

2、合并单机多卡timeline，只合并0卡和1卡：

```
python3 main.py -d path/to/cann_profiling/ --rank 0,1
```

3、合并单机多卡timeline，合并所有卡的CANN层和Ascend_Hardware层数据
```
python3 main.py -d path/to/cann_profiling/ --items CANN,Ascend_Hardware
```

4、合并多机多卡的timeline时, 需要-t指定节点间的时间误差文件路径, 用以校准节点间的时间：

```
python3 main.py -d path/to/cann_profiling/ -t path/to/time_difference.json --rank 0,8,
```

合并timeline查看：
> 在 -o 指定的目录（默认在-d指定的目录下）的msprof_merged_*p.json为合并后的文件

## 2 多timeline融合(自定义)
### 2.1 数据采集
将需要合并的timeline文件全部放在同一目录下
数据目录结构示意, 合并timeline必需数据：`msprof.json`和`info.json.*`：
```
|- timeline
    |- msprof_0.json
    |- msprof_1.json
    |- msprof_2.json
    |- msprof_3.json
    |- step_trace_0.json
    |- step_trace_1.json
    |- step_trace_2.json
    |- step_trace_3.json
    ...
```
### 2.2 合并timeline
使用脚本`merge_profiling_timeline/main.py`合并timeline

可选参数：
- -d: **必选参数**，指定profiling数据文件或文件夹路径
- -o: 可选参数，指定合并后的timeline文件输出的路径，默认为'-d'输入的路径
- --custom: **必选参数**，工具通过该参数识别为自定义融合场景
**使用示例**：

将需要合并的所有timeline放在同一目录下，通过下面的命令合并所有timeline
```
python3 main.py -d path/to/timeline/ --custom
```
合并timeline查看：同

## 3 超大timeline文件查看

直接使用以下命令
```
cd merge_profiling_timeline
python ./trace_processor --httpd path/to/msprof_merged_*p.json 
```
等待加载完毕，刷新[perfetto](https://ui.perfetto.dev/)界面，点击`YES, use loaded trace`即可展示timeline



