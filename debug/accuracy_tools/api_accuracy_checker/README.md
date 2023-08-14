# Ascend模型精度预检工具
Ascend模型精度预检工具能在昇腾NPU上扫描用户训练模型中所有API，给出它们精度情况的诊断和分析。工具会提取模型中所有的API前反向的信息，构造相应的API单元测试，将NPU输出与标杆比对，从而检测出精度有问题的API。

## 工具特性
1. 落盘数据小
2. 不依赖标杆侧GPU训练资源，本地即可完成预检
3. 支持随机生成模式和真实数据模式
4. 单API测试，排除整网中的累计误差问题

## 使用方式

1. 安装预检工具

   将att仓代码下载到本地，并配置环境变量。假设下载后att仓路径为 $ATT_HOME，环境变量应配置为

   ```
   export PYTHONPATH=$PYTHONPATH:$ATT_HOME/debug/accuracy_tools/
   ```
   安装依赖tqdm
   ```
   pip3 install tqdm
   ```
   安装依赖tqdm
   ```
   pip install tqdm
   ```

2. 在训练脚本（如main.py）中加入以下代码导入工具dump模块，启动训练即可自动抓取网络所有API信息

   ```
   import api_accuracy_checker.dump
   ```

   目前工具仅支持抓取训练的**第二个迭代**并且在第二个迭代后会报错退出训练进程。报错信息如下，这个报错仅用于停止训练，属于正常现象：
   ```
   Exception: Model pretest: exit after iteration 1.
   ```

​	dump信息默认会存盘到'./'路径下（相对于启动训练的路径），包括前向API信息forward_info_{pid}.json, 反向API信息backward_info_{pid}.json, 调用栈信息stack_info_{pid}.json。真实数据模式下还有forward_real_data和backward_real_data文件夹，里面有每个api输入的具体数值。forward_info与stack_info中的key值一一对应，用户可根据forward_info中API的key在stack_info中查询到其调用栈及代码行位置。

   有需要的话，用户可以通过msCheckerConfig.update_config来配置dump路径以及启用真实数据模式（默认为关）。注意启用真实数据模式目前仅支持单卡，且会存盘较多数据，可能对磁盘空间有较大冲击。
   ```
      from api_accuracy_checker.dump import msCheckerConfig
      msCheckerConfig.update_config(dump_path="my/dump/path", real_data=True)
   ```

"my/dump/path" 需配置为用户想要的api信息存盘路径，并且需要提前创建好

3. 将上述信息输入给run_ut模块运行精度检测并比对，运行如下命令：

   ```
   cd $ATT_HOME/debug/accuracy_tools/api_accuracy_checker/run_ut
   python run_ut.py -forward ./forward_info_0.json -backward ./backward_info_0.json
   ```

   forward和backward两个命令行参数根据实际存盘的json文件名配置。比对结果存盘路径默认是'./'（相对于run_ut的路径），可以在运行run_ut.py时通过 --out_path命令行参数配置。结果包括pretest_result.csv和pretest_details.csv两个文件。前者是api粒度的，标明每个api是否通过测试。建议用户先查看前者，对于其中没有通过测试的或者特定感兴趣的api，根据其API name字段在pretest_details.csv中查询其各个输出的达标情况以及比较指标。

   注意：目前API通过测试的标准是每个输出与标杆比对的余弦相似度大于0.99，pretest_details.csv中的相对误差供用户分析时使用。









