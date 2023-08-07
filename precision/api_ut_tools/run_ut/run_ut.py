# 用户构造并运行api用例，注意前反向的区分
import yaml
import os
import json
import torch

FLOAT_TYPE = ['torch.float32', 'torch.float', 'torch.float64', 'torch.double', 'torch.float16', \
            'torch.half', 'torch.bfloat16']

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapFunctionalOps = yaml.safe_load(f).get('functional')

for f in dir(torch.nn.functional):
    locals().update({f: getattr(torch.nn.functional, f)})


def run_ut():
    print("start")
    forward_pkl = open("/home/wangchao/torch_test/dump_data_new/npu/ptdbg_dump_v1.0/rank0/dump.pkl")
    backward_pkl = open("/home/wangchao/torch_test/dump_data_new/npu/ptdbg_dump_v1.0/rank0/dump_backward.pkl")
    forward_content = forward_pkl.readlines()
    backward_content = backward_pkl.readlines()
    for api_info in forward_content:
        api_json = json.loads(api_info)
        for key, value in api_json.items():
            [api_type, api_name, index, mode] = key.split("*")
            print(api_name)
            api_feature = key.rsplit("*", 1)[0]
            args, kwargs = generate_input(value.get("args"), api_json.get("kwargs"))
            if api_type == "Functional":
                out = eval(api_name)(*args, **kwargs)
            if api_type == "Tensor":
                out = getattr(torch._C._TensorBase, str(api_name))(*args, **kwargs)
            for line in backward_content:
                if api_feature in line:
                    api_back_json = json.loads(line)
                    for params in api_back_json.values():
                        grad = nested_generate_input(params.get("args"), True, False)
                        out.backward(grad)
            input_grad = [tensor.grad for tensor in args if isinstance(tensor, torch.Tensor)]
            print("forward")
            print(out)
            print("backward")
            print(input_grad)


def generate_input(input_args, input_kwargs, need_backward=True, need_convert=False):  # 没有考虑dict of tensor
    args = []
    kwargs = {}

    for info in input_args:
        args.append(nested_generate_input(info, need_backward, need_convert))
    if kwargs:
        for key, info in input_kwargs.items():
            kwargs[key] = nested_generate_input(info, need_backward, need_convert)
    return args, kwargs


def nested_generate_input(info, need_backward, need_convert):
    if isinstance(info, list):
        result = []
        for i in info:
            result.append(nested_generate_input(i, need_backward, need_convert))
        return result
        # return list(map(nested_generate_input, info))
    # elif isinstance(input_info, tuple):
    #     return tuple(map(generate_input, input_info))
    else:
        if info['type'] == 'torch.Tensor':
            low, high = info['Min'], info['Max']
            data_dtype = info['dtype']
            if data_dtype in FLOAT_TYPE: #应该搞个float类型列表
                if need_convert and data_dtype == "torch.float16":
                    data_dtype = "torch.float32"
                scale = high - low
                rand01 = torch.rand(tuple(info['shape']), dtype=eval(data_dtype))
                inpt = rand01 * scale + low
                if need_backward:
                    inpt.requires_grad_(True)
                    inpt.retain_grad()
            elif 'int' in data_dtype or 'long' in data_dtype: # 应该搞个int类型列表,
                inpt = torch.randint(int(low), int(high)+1, tuple(info['shape']),
                                     dtype=eval(data_dtype)) # high + 1因为右边是开区间
            else:
                print(f'Warning: Dtype is not supported: ', info['dtype'])
                raise NotImplementedError()
        else:
            inpt = info['value'] # 遗留问题：需要考虑是否要转换成原本类型
        return inpt

run_ut()