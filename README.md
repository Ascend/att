# att

#### 介绍
Ascend Training Tools，昇腾训练工具链
针对训练&大模型场景，提供端到端命令行&可视化调试调优工具，帮助用户快速提高模型开发效率

#### 模型训练迁移全流程
![输入图片说明](debug/resources/model_training_migration_process.png)

#### 使用说明
1.  性能工具[tools](https://gitee.com/ascend/att/tree/master/profiler)
    a. [compare_tools](https://gitee.com/ascend/att/tree/master/profiler/compare_tools)

        **GPU与NPU性能比较工具**：提供NPU与GPU性能拆解功能以及算子、通信、内存性能的比较功能。

    b. [distribute_tools](https://gitee.com/ascend/att/tree/master/profiler/distribute_tools)

        **集群场景脚本集合**：提供集群场景数据一键汇聚功能。

    c. [merge_profiling_timeline](https://gitee.com/ascend/att/tree/master/profiler/merge_profiling_timeline)

        **合并大json工具**：融合多个profiling的timeline在一个json文件中的功能。

    d. [cluster_analyse](https://gitee.com/ascend/att/tree/master/profiler/cluster_analyse)
        **集群分析工具**：提供多机多卡的集群分析能力（基于通信域的通信分析和迭代耗时分析）, 当前需要配合Ascend Insight的集群分析功能使用。


2.  精度工具[tools](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools)

    a. [api_accuracy_checker](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools/api_accuracy_checker)

        **预检功能**：Ascend模型精度预检工具能在昇腾NPU上扫描用户训练模型中所有API，给出它们精度情况的诊断和分析。

    b. [ptdbg_ascend](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools/ptdbg_ascend)

        **PyTorch精度工具**：用来进行PyTorch整网API粒度的数据dump、精度比对和溢出检测，从而定位PyTorch训练场景下的精度问题。

    
3.  tensorboard支持npu可视化插件[tb-plugin](https://gitee.com/ascend/att/tree/master/plugins/tensorboard-plugins/tb_plugin)

    **PyTorch profiling数据可视化的TensorBoard的插件**： 它支持将Ascend平台采集、解析的Pytorch Profiling数据可视化呈现，也兼容GPU数据采集、解析可视化。

#### 参与贡献

1.  Fork 本仓库
2.  新建 xxx 分支
3.  提交代码
4.  新建 Pull Request

