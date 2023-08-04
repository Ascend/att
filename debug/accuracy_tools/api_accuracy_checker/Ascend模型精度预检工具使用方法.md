# Ascend模型精度预检工具

## 使用方式

1. 安装遇见工具

   将att仓代码下载到本地，并配置环境变量。假设att仓本地路径为 {att_root}，环境变量应配置为

   ```
   export PYTHONPATH=$PYTHONPATH:{att_root}/debug/accuracy_tools/
   ```

2. 使用工具dump模块抓取网络所有API信息

   ```
   from api_accuracy_checker.dump import set_dump_switch
   set_dump_switch("ON")
   ```

​	dump信息默认会存盘到./api_info/路径下，后缀的数字代表进程pid

3. 将上述信息输入给run_ut模块运行精度检测并比对

   ```
   cd run_ut
   python run_ut.py --forward ./api_info/forward_info_0.json --backward ./api_info/backward_info_0.json
   ```

   forward和backward两个命令行参数根据实际情况配置。比对结果存盘位置会打屏显示，默认是'./'，可以在运行run_ut.py时通过 --out_path命令行参数配置。

