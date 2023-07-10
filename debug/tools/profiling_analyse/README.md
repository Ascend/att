# profiling_analyse

## gpu性能数据拆解
### 算子耗时
1. 获取json文件中的traceEvents字段，获取所有cat为kernel的，且name中不包含nccl的event，将他们的耗时相加即为所有算子耗时all_time；
2. 取算子中name包含gemm的为cube算子耗时cube_time.
3. vector算子耗时即为(all_time - cube_time)
### 通信
此处的通信指通信未掩盖耗时，通过计算有通信流而无计算流的时间戳获得
### 计算流e2e耗时
按照时间也就是'ts'字段排序所有events，可以看到最后的event是Record Window End,故使用最后一个event的ts值减去第一个event的ts值作为计算流e2e耗时
### 调度
gpu上的调度耗时计算方法采用：调度耗时 = 单步打屏时间 - 算子耗时 - 通信不可掩盖耗时
单步打屏时间需要用户输入，当用户不输入时，采用e2e耗时代替单步打屏时间
获得调度耗时后，使用调度占比 = 调度耗时/E2E耗时 获取调度占比
### 内存分析
gpu上的内存使用可以使用nvidia-smi查看，使用json文件分析时需要打开profile_memory=True开关
获取运行稳定后的step的profiling数据，获取所有name值为[memory]的event，获取这类event中Total Reserved值的最大值作为内存值

## npu性能数据解析
### 算子耗时
1、算子耗时profiling数据位于/PROFxxx/device_x/summary路径下的op_summary_x_1.csv文件中。
2、当前仅统计算子运行在vector和cube上的耗时。
3、这2中算子于csv文件中的的TaskType均为AI_CORE，其中aiv_vec_time时间多表明为vector算子，aic_mac_time表明为cube算子。分别累加求和算子耗时进行输出。
4、算子若无pmu信息，仅统计ai_core的总耗时并显示在结果"vector算子"一列

### 通信
1、此处的通信为通信未掩盖耗时，对应为ASCEND_PROFILER_OUTPUT/trace_view.json下同一条流的EVENT_WAIT_SQE总耗时。
2、选取trace_view中的计算流——即流中同时存在EVENT_WAIT_SQE和Task Type为AI_CORE的流
3、对于AI_CORE存在2条流中的情况，计算流中累加EVENT_WAIT_SQE时会减去同时间区间内另外流产生的AI_CORE耗时

### 计算流e2e耗时
此耗时通过统计trace_view.json中compute_time时间戳‘ts’的最小值和最大值，其时间差的绝对值即为e2e耗时。

### 调度占比
1、调度占比的求取需先计算调度耗时，调度占比=调度耗时/e2e耗时 * 100%。
2、调度耗时的计算方法有2种，①调度耗时=单步打屏时间-算子耗时-通信不可掩盖耗时，②调度耗时=e2e耗时-计算流执行任务总耗时。
3、由于”单步打屏时间“需额外记录输入，增加可选输入字段“-nlt”，作为用户的可选输入“单步打屏时间”，若无输入，该值使用e2e耗时替代。

### 内存
1、内存统计的数据来源于ASCEND_PROFILER_OUTPUT/memory_record.csv中的”Total Reserved(MB)“。
2、其值在模型训练趋于稳定时逐渐固定，整体偏差不大，因此输出结果为该列数据的最大值。

## 样例
- step1:获取gpu和npu的profiling数据，若采集profiling数据时没开启memory采集开关，则没有内存使用数据

- 运行命令:python profiling_parse.py -g gpu\gpu_trace_device0.json -glt 0.9 -n npu\xxx_ascend_pt -nlt 1.2
- 输出结果：可以得到gpu与npu对照的打屏性能拆解数据
