## FAQ
## 工具使用
### 1.环境变量方式导入ptdbg_ascend
当需要使用export att/debug/accuracy_tools/ptdbg_ascend/src/python/ptdbg_ascend/common的目录下，手动添加一个version.py，并加上以下版本号信息，其中‘3.4’为当前ptdbg_ascend的版本

```
__version__ = '3.4'
```
### 2.dump指定融合算子
dump指定操作当前支持dump指定融合算子的输入输出，需要在att/debug/accuracy_tools/ptdbg_ascend/src/python/ptdbg_ascend/hook_module/support_wrap_ops.yaml中添加，比如以下算子：
```
def npu_forward_fused_softmax(self, input_, mask):
    resl = torch_npu.npu_scaled_masked_softmax(input_, mask, self.scale, False)
    return resl
```
如果需要dump其中调用的npu_scaled_masked_softmax算子的输入输出信息，需要在support_wrap_ops.yaml中的torch_npu: 中自行添加该融合算子即可：
```
- npu_scaled_masked_softmax
```
（仅作举例用，已默认支持）

## 常见问题

### 1. 在同一个目录多次执行dump会冲突吗？

会，同一个目录多次dump，会覆盖上一次结果，可以使用dump_tag参数修改dump目录名称。

### 2. 一个网络中包含多个model，register hook中传入哪一个model？

传入任意一个model即可，工具会自动dump所有model。

### 3. 如何dump算子级的数据？

需要使用acl dump模式，即在dump操作中配置mode="acl"或dump_mode='acl'。

### 4. 工具比对发现NPU和标杆数据的API无法完全对齐？

torch版本和硬件差异属于正常情况

## 异常情况
### 1. 单机多卡场景dump目录下只生成一个rank目录或pkl文件格式损坏

**故障现象**

dump目录下只生成一个rank目录或dump目录下的pkl文件格式损坏、内容不完整。 

**故障原因**

通常是因为register_hook没有正确配置，带着工具没有获取正确的`rank_id`（从rank参数读取或从模型参数的device_id读取）。

**故障处理**

register_hook需要在set_dump_path之后调用，也需要在每个进程上被调用，建议在搬运模型数据到卡之后调用。识别方法如下：

- 找到训练代码中遍历epoch的for循环或遍历数据集的for循环，把register_hook放到循环开始前即可。
- 找到训练代码中调用DDP或者DistributedDataParallel的代码行，把register_hook放到该代码行所在的代码块之后。
- 若代码中均无以上两种情况，那么尽可能把这行代码往后放，并配置register_hook的rank参数。

### 2. HCCL 报错： error code: EI0006

**故障现象**

使用ptdbg_ascend工具时，报错： error code: EI0006。

**故障原因**

CANN软件版本较低导致不兼容。

**故障处理**

升级新版CANN软件版本。
### 3. torch_npu._C._clear_overflow_npu() RuntimeError NPU error，error code is 107002
如果运行溢出检测功能遇到这个报错，采取以下解决方法：
如果是单卡运行，添加如下代码，0是卡号，选择自己空闲的卡号。

```
torch.npu.set_device('npu:0')
```
如果多卡运行，请在代码中修改对应卡号，比如进程使用卡号为{rank}时可以添加如下代码：

```
torch.npu.set_device(f'npu:{rank}')
```
如果运行精度比对功能遇到这个报错，尝试安装最新版本的ptdbg_ascend

### 4. 运行compare.py时报错：json.decoder.JSONDecodeError: Extra data: line 1 column 37(char 36)

遇到这种情况，先更新工具版本为最新版本，再重新运行训练代码dump数据，再用新的dump数据进行精度比对，如果最新版本未能解决问题，请联系ptdbg工具开发人员。

### 5. AssertionError: assert set(WrapTensorOps) <= set(_tensor_ops)

遇到这种情况，先检查安装的torch版本，建议先更新工具版本为2.2以上，版本2.2的工具支持torch1.8、1.11和2.0

### 6. dump得到的VF_lstm_99_forward_input.1.0.npy、VF_lstm_99_forward_input.1.1.npy类似的数据是否正常？

带1.0/1.1/1.2后缀的npy是正常现象，例如当输入数据为[[tensor1, tensor2, tensor3]]会生成这样的后缀

### 7. dump数据时，dump输出目录只得到了.npy文件，不生成pkl文件

- 检查set_dump_switch("ON")，set_dump_switch("OFF")是否都配置了；
- 如果都配置了，观察模型运行日志结尾是否打印“Dump switch is turned off”，如果没有，则表明代码没有执行到set_dump_switch("OFF")，请检查模型代码中是否有exit()操作。

### 8. 进行compare报错：The current file contains stack information, please turn on the stack_mode
在比对脚本中，设置stack_mode=True，例如：

```
from ptdbg_ascend import *
dump_result_param={
"npu_pkl_path": "./npu_dump/ptdbg_dump_v2.0/rank0/api_stack_dump.pkl",
"bench_pkl_path": "./gpu_dump/ptdbg_dump_v2.0/rank0/api_stack_dump.pkl",
"npu_dump_data_dir": "./npu_dump/ptdbg_dump_v2.0/rank0/api_stack_dump",
"bench_dump_data_dir": "./gpu_dump/ptdbg_dump_v2.0/rank0/api_stack_dump",
"is_print_compare_log": True
}
compare(dump_result_param, "./output", stack_mode=True)
```
### 9. dump指定反向API的ACL级别的数据报错：NameError：name 'torch_npu' is not defined

- 如果是npu环境，请安装torch_npu；
- 如果是gpu环境，暂不支持dump指定API的ACL级别的数据

### 10. 配置dump_path后，使用工具报错：[ERROR]The file path /home/xxx/dump contains special characters

- 请检查你设置的dump绝对路径是否包含特殊字符，确保路径名只包含大小写字母、数字、下划线、斜杠、点和短横线
- 注意，如果你执行脚本的路径为/home/abc++/，你设置的dump_path="./dump"，工具实际校验的路径为绝对路径/home/abc++/dump，++为特殊字符，会引发本条报错

### 11. 遇到报错'IsADirectoryError: [Errno 21] Is a directory: '/data/rank0/api_stack_xxx''

- 请检查register_hook是否写在了set_dump_path前面，register_hook必须在set_dump_path后调用
- 请检查是否写了多个register_hook或者set_dump_path，如有，请保留一个register_hook或者set_dump_path

### 12. 无法dump matmul权重的反向梯度数据

- matmul期望的输入是二维，当输入不是二维时，会将输入通过view操作展成二维，再进行matmul运算，因此在反向求导时，backward_hook能拿到的是UnsafeViewBackward这步操作里面数据的梯度信息，取不到MmBackward这步操作里面数据的梯度信息，即权重的反向梯度数据。
- 典型的例子有，当linear的输入不是二维，且无bias时，会调用output = input.matmul(weight.t()),因此拿不到linear层的weight的反向梯度数据。

### 13. pkl文件中的某些api的dtype类型为float16，但是读取此api的npy文件显示的dtype类型为float32

- ptdbg工具在dump数据时需要将原始数据从npu to cpu上再转换为numpy类型，npu to cpu的逻辑和gpu to cpu是保持一致的，都存在dtype可能从float16变为float32类型的情况，如果出现dtype不一致的问题，最终dump数据的dtype以pkl文件为准。

### 14. 使用dataloader后raise异常Exception: ptdbg: exit after iteration [x, x, x]

- 正常现象，dataloader通过raise结束程序，堆栈信息可忽略。

### 15. 工具报错：AssertionError: Please register hooks to nn.Module

- 请在model示例化之后配置register hook。

### 16. 添加ptdbg_ascend工具后截取操作报错：`IndexError: too many indices for tensor of dimension x` 或 `TypeError: len() of a 0-d tensor`。

删除ptdbg_ascend工具的hook_module目录下yaml文件中Tensor:下的`- __getitem__`即可。
