# Ascend模型精度预检工具

Ascend模型精度预检工具能在昇腾NPU上扫描用户训练模型中所有API，输出精度情况的诊断和分析。工具会提取模型中所有的API前反向信息，构造相应的API单元测试，将NPU输出与标杆比对，从而检测出精度有问题的API。

工具支持PyTorch版本：1.8.1/1.11.0/2.0/2.1。

## 工具特性

1. 落盘数据小
2. 不依赖标杆侧GPU训练资源，本地即可完成预检
3. 支持随机生成模式和真实数据模式
4. 单API测试，排除整网中的累计误差问题

## 使用方式

1. 安装预检工具

   将att仓代码下载到本地，并配置环境变量。假设下载后att仓路径为 $ATT_HOME，环境变量应配置为：

   ```bash
   export PYTHONPATH=$PYTHONPATH:$ATT_HOME/debug/accuracy_tools/
   ```

   安装依赖tqdm、prettytable、pyyaml

   ```bash
   pip3 install tqdm prettytable pyyaml
   ```

2. 在训练脚本（如main.py）中加入以下代码导入工具dump模块，启动训练即可自动抓取网络所有API信息

   ```python
   import api_accuracy_checker.dump
   ```

   若训练脚本中的代码不是通过dataloader来加载数据或在部分流水并行、张量并行场景下，工具的开关无法在每张卡上自动打开，导致多卡训练dump结果只有一组json，那么需要在训练代码中添加打开工具开关的调用：

     ```Python
   import api_accuracy_checker.dump as DP
   DP.dump.set_dump_switch("ON")
     ```

   上述代码要添加在迭代前向的代码段中，或者说是遍历数据集循环的代码段中。如对于GPT-3可以添加在pretrain_gpt.py 的forward_step函数中。之后工具会适配这个场景开关的自动打开。

   工具默认抓取训练的**第二个迭代**并且在第二个迭代后会报错退出训练进程，可通过target_iter参数配置。报错信息如下，这个报错仅用于停止训练，属于正常现象：

   ```bash
   Exception: Model pretest: exit after iteration 1.
   ```

   dump信息默认会存盘到“./”路径下（相对于启动训练的路径），包括：

   - forward_info_{pid}.json：前向API信息文件。
   - backward_info_{pid}.json：反向API信息文件。
   - stack_info_{pid}.json：调用栈信息文件。

   forward_info与stack_info中的key值一一对应，用户可根据forward_info中API的key在stack_info中查询到其调用栈及代码行位置。

   若有需要，用户可以通过msCheckerConfig.update_config来配置dump路径以及开启真实数据模式，在训练脚本中加入如下示例代码：

   ```Python
   	from api_accuracy_checker.dump import msCheckerConfig
   	msCheckerConfig.update_config(dump_path="my/dump/path", real_data=True, enable_dataloader=True, target_iter=1)
   ```

   | 参数名称          | 说明                                                         | 是否必选 |
   | ----------------- | ------------------------------------------------------------ | -------- |
   | dump_path         | 设置dump路径，须为已存在目录，默认为当前目录。               | 否       |
   | real_data         | 真实数据模式，可取值True或False，默认为False，配置为True后开启真实数据模式，dump信息增加forward_real_data和backward_real_data目录，目录下保存每个API输入的具体数值。开启真实数据模式目前仅支持单卡，且会存盘较多数据，可能对磁盘空间有较大冲击。 | 否       |
   | enable_dataloader | 自动控制开关，可取值True或False，默认为True，配置为True后自动识别dump target_iter参数指定的迭代数据，并在该迭代执行完成后退出训练。 | 否       |
   | target_iter       | 指定dump某个step的数据，默认为1，仅支持dump1个step，须指定为训练脚本中存在的step。 | 否       |

3. 将API信息输入给run_ut模块运行精度检测并比对，运行如下命令：

   ```bash
   cd $ATT_HOME/debug/accuracy_tools/api_accuracy_checker/run_ut
   python run_ut.py -forward ./forward_info_0.json -backward ./backward_info_0.json
   ```

   | 参数名称         | 说明                                                         | 是否必选 |
   | ---------------- | ------------------------------------------------------------ | -------- |
   | -forward         | 指定前向API信息文件forward_info_{pid}.json。                 | 是       |
   | -backward        | 指定反向API信息文件backward_info_{pid}.json。                | 是       |
   | -save_error_data | 保存精度未达标的API输入输出数据。                            | 否       |
   | --out_path       | 指指定run_ut执行结果存盘路径，默认“./”（相对于run_ut的路径）。 | 否       |

   run_ut执行结果包括accuracy_checking_result.csv和accuracy_checking_details.csv两个文件。accuracy_checking_result.csv是API粒度的，标明每个API是否通过测试。建议用户先查看accuracy_checking_result.csv文件，对于其中没有通过测试的或者特定感兴趣的API，根据其API name字段在accuracy_checking_details.csv中查询其各个输出的达标情况以及比较指标。

   注意：目前API通过测试的标准是每个输出与标杆比对的余弦相似度大于0.99，并且float16和bfloat16数据要通过双千分之一标准，float32数据要通过双万分之一标准，accuracy_checking_details.csv中的相对误差供用户分析时使用。

4. 如果需要保存比对不达标的输入和输出数据，可以在run_ut执行命令结尾添加-save_error_data，例如：

   ```bash
   python run_ut.py -forward ./forward_info_0.json -backward ./backward_info_0.json -save_error_data
   ```
   数据默认会存盘到'./ut_error_data'路径下（相对于启动run_ut的路径），有需要的话，用户可以通过msCheckerConfig.update_config来配置保存路径，参数为error_data_path

# 溢出API解析工具

针对训练过程中的溢出检测场景，对于输入正常但输出存在溢出的API，会在训练执行目录下将溢出的API信息按照前向和反向分类，dump并保存为`forward_info_{pid}.json`和`backward_info_{pid}.json`，前向过程溢出的API可通过该工具对`forward_info_{pid}.json`进行解析，输出溢出API为正常溢出还是非正常溢出，从而帮助用户快速判断。

工具支持PyTorch版本：1.8.1/1.11.0/2.0/2.1。

参见[ptdbg_ascend精度工具功能说明](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools/ptdbg_ascend/doc)中的"溢出检测场景"进行溢出检测dump。

若dump结果生成`forward_info_{pid}.json`和`backward_info_{pid}.json`文件，则使用本工具进行解析。操作步骤如下：

1. 安装预检工具

   将att仓代码下载到本地，并配置环境变量。假设下载后att仓路径为 $ATT_HOME，环境变量应配置为

   ```bash
   export PYTHONPATH=$PYTHONPATH:$ATT_HOME/debug/accuracy_tools/
   ```

   安装依赖tqdm、prettytable、pyyaml

   ```bash
   pip3 install tqdm prettytable pyyaml
   ```

2. 执行溢出API解析操作

   ```bash
   cd $ATT_HOME/debug/accuracy_tools/api_accuracy_checker/run_ut
   python run_overflow_check.py -forward ./forward_info_0.json
   ```

   反向过程溢出的API暂不支持该功能。


具体参数解释请参见“Ascend模型精度预检工具”。

# FAQ 

1. run ut过程中出现报错：ERROR:Got unsupported ScalarType BFloat16

   答：请使用最新版本的工具

2. Dropout算子，CPU和NPU的随机应该不一样，为什么结果比对是一致的？

   答：这个结果是正常的，工具对该算子有特殊处理，只判定位置为0的位置比例大约和设定p值相当

3. 为什么浮点型数据bench和npu的dtype不一致？

   答：对于fp16的数据，cpu会上升一个精度fp32去计算，这是和算子那边对齐的精度结论，cpu用更高精度去计算会更接近真实值
