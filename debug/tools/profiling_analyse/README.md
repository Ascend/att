# profiling_analyse

## gpu性能数据拆解
### 算子耗时
1. 获取json文件中的traceEvents字段，获取所有cat为kernel的，且name中不包含nccl的event，将他们的耗时相加即为所有算子耗时all_time；
2. 取算子中name包含gemm的为cube算子耗时cube_time.
3. vector算子耗时即为(all_time - cube_time)
### 大kernel算子
待补充大kernel列表
### 通信
此处的通信指通信未掩盖耗时，gpu上暂无明确的计算方法，故取的profiling展示图中的通信流耗时结果
实际计算是取name中包含'ncclKernel_'的event，将他们的耗时相加
### 计算流e2e耗时
按照时间也就是'ts'字段排序所有events，可以看到最后的event是Record Window End,故使用最后一个event的ts值减去第一个event的ts值作为计算流e2e耗时
### 调度
由于gpu上目前没有较好的通信不可掩盖耗时算法，所以gpu上的调度耗时计算方法采用：调度耗时 = 计算流E2E耗时 - 计算流任务执行总耗时
计算流为stream为7的流，实际计算取stream为7的event耗时相加；
计算流的stream不一定是7，后续会适配通过观察kernel算子分布来判断计算流的方法
获得调度耗时后，使用调度占比 = 调度耗时/E2E耗时 获取调度占比
### 内存分析
gpu上的内存使用可以使用nvidia-smi查看，使用json文件分析时需要打开profile_memory=True开关
获取运行稳定后的step的profiling数据，获取所有name值为[memory]的event，获取这类event中Total Reserved值的最大值作为内存值

## npu性能数据解析
### 算子耗时
1、算子耗时profiling数据位于/PROFxxx/device_x/summary路径下的op_summary_x_1.csv文件中。
2、当前仅统计算子运行在vector和cube上的耗时。、
3、这2中算子于csv文件中的的TaskType均为AI_CORE，其中aiv_vec_time时间多表明为vector算子，aic_mac_time表明为cube算子。分别累加求和算子耗时进行输出。

### 大kernel算子
待补充大kernel算子列表

### 通信
此处的通信为通信未掩盖耗时，对应为ASCEND_PROFILER_OUTPUT/trace_view.json下的communication_not_overlapped。
输出结果为该字段时间求和。

### 计算流e2e耗时
此耗时通过统计trace_view.json中时间戳‘ts’的最小值和最大值，其时间差的绝对值即为e2e耗时。

### 调度占比
1、调度占比的求取需先计算调度耗时，调度占比=调度耗时/e2e耗时 * 100%。
2、调度耗时的计算方法有2种，①调度耗时=单步打屏时间-算子耗时-通信不可掩盖耗时，②调度耗时=e2e耗时-计算流执行任务总耗时。
3、由于”单步打屏时间“需额外记录输入，暂不使用方法①，方法②中的计算流执行任务总耗时即为trace_view.json下的compute_time。

### 内存
1、内存统计的数据来源于ASCEND_PROFILER_OUTPUT/memory_record.csv中的”Total Reserved(MB)“。
2、其值在模型训练趋于稳定时逐渐固定，整体偏差不大，因此输出结果为该列数据的最大值。

## 样例
- step1:下载数据：https://onebox.huawei.com/v/2ad3400460fac22fa61f21f478edd116

- 运行命令:python profiling_parse.py -g prof0704_best\gpu\gpu_trace_device0.json -n prof0704_best\Malluma_443350_20230704144255_ascend_pt
- 输出结果：可以得到gpu与npu对照的打屏性能拆解数据
