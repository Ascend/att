#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2019-2020. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import inspect
import json
import os
import stat
import numpy as np
import torch
import threading

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False

from .utils import DumpUtil, _set_dump_switch4api_list, make_dump_data_dir, get_tensor_rank, create_dirs_if_not_exist
from ..common.utils import print_warn_log, Const, print_info_log, modify_dump_path
from ..dump.utils import check_writable

forward_init_status = False
backward_init_status = False

backward_threading_id = 0

api_list = []
thread_lock = threading.Lock()
pkl_name = ""
multi_output_apis = ["_sort_", "npu_flash_attention"]

class DataInfo(object):
    def __init__(self, data, save_data, summary_data, dtype, shape):
        self.data = data
        self.save_data = save_data
        self.summary_data = summary_data
        self.dtype = dtype
        self.shape = shape


def get_not_float_tensor_info(data):
    if data.numel() == 0 or data.dtype == torch.bool:
        tensor_max = []
        tensor_min = []
        tensor_mean = []
    elif len(data.shape) == 0:
        tensor_max = data.cpu().detach().float().numpy().tolist()
        tensor_min = data.cpu().detach().float().numpy().tolist()
        tensor_mean = data.cpu().detach().float().numpy().tolist()
    else:
        tensor_max = torch._C._VariableFunctionsClass.max(data).cpu().detach().float().numpy().tolist()
        tensor_min = torch._C._VariableFunctionsClass.min(data).cpu().detach().float().numpy().tolist()
        tensor_mean = torch._C._VariableFunctionsClass.mean(data.float()).cpu().detach().float().numpy().tolist()
    return get_tensor_data_info(data, tensor_max, tensor_min, tensor_mean)


def get_scalar_data_info(data):
    summary_data = [data, data, data]
    return DataInfo(data, data, summary_data, str(type(data)), str([]))


def get_float_tensor_info(data):
    tensor_max = torch._C._VariableFunctionsClass.max(data).cpu().detach().float().numpy().tolist()
    tensor_min = torch._C._VariableFunctionsClass.min(data).cpu().detach().float().numpy().tolist()
    tensor_mean = torch._C._VariableFunctionsClass.mean(data).cpu().detach().float().numpy().tolist()
    return get_tensor_data_info(data, tensor_max, tensor_min, tensor_mean)


def get_tensor_data_info(data, tensor_max, tensor_min, tensor_mean):
    summary_data = []
    saved_tensor = data.contiguous().cpu().detach()
    if data.dtype == torch.bfloat16:
        saved_numpy = saved_tensor.to(torch.float32).numpy()
    else:
        saved_numpy = saved_tensor.numpy()
    summary_data.extend([tensor_max, tensor_min, tensor_mean])
    return DataInfo(data, saved_numpy, summary_data, str(data.dtype), tuple(data.shape))


def json_dump_condition(prefix):
    cur_threading_id = threading.current_thread().ident
    global backward_threading_id
    if not backward_threading_id and Const.BACKWARD in prefix:
        backward_threading_id = cur_threading_id
    return (Const.BACKWARD in prefix and backward_threading_id == cur_threading_id) or 'forward' in prefix


def dump_tensor(x, prefix, dump_step, dump_file_name):
    global data_info
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            dump_tensor(item, "{}.{}".format(prefix, i), dump_step, dump_file_name)
        return
    elif isinstance(x, torch.Tensor):
        if x.is_meta:
            print_info_log(f"Meta tensor {prefix} is skipped.")
            return
        if x.numel() == 0 or len(x.shape) == 0 or not x.is_floating_point():
            if DumpUtil.dump_filter_switch == Const.OFF:
                data_info = get_not_float_tensor_info(x)
                dump_data(dump_file_name, dump_step, prefix, data_info)
            else:
                return
        else:
            data_info = get_float_tensor_info(x)
            dump_data(dump_file_name, dump_step, prefix, data_info)

    elif DumpUtil.dump_filter_switch == Const.OFF:
        if isinstance(x, bool) or isinstance(x, int) or isinstance(x, float):
            data_info = get_scalar_data_info(x)
            dump_data(dump_file_name, dump_step, prefix, data_info)


def dump_data(dump_file_name, dump_step, prefix, data_info):
    global api_list
    thread_lock.acquire()
    try:
        if json_dump_condition(prefix):
            output_path = os.path.join(DumpUtil.dump_data_dir, f'{prefix}.npy')
            if not DumpUtil.summary_only:
                np.save(output_path, data_info.save_data)
            api_list.append([prefix, dump_step, [], data_info.dtype, data_info.shape, data_info.summary_data])
    except Exception as e:
        print_warn_log("Dump data failed, error: {}".format(e))
    finally:
        thread_lock.release()


def dump_stack_info(name_template, dump_file):
    stack_str = []
    for (_, path, line, func, code, _) in inspect.stack()[3:]:
        if code:
            stack_line = [path, str(line), func, code[0].strip() if code else code]
        else:
            stack_line = [path, str(line), func, code]
        stack_str.append(stack_line)

    prefix = name_template.format("stack_info")
    if DumpUtil.dump_switch_mode in Const.DUMP_MODE:
        if json_dump_condition(prefix):
            if Const.ALL in DumpUtil.dump_mode:
                api_list.append([prefix, stack_str])
            else:
                for mode in DumpUtil.dump_mode:
                    if mode in prefix:
                        api_list.append([prefix, stack_str])
    else:
        api_list.append([prefix, stack_str])


def dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file):
    if Const.BACKWARD in name_template and Const.FORWARD not in DumpUtil.dump_mode:
        if 'input' in DumpUtil.dump_mode:
            dump_tensor(out_feat, name_template.format("input"), dump_step, dump_file)
        if 'output' in DumpUtil.dump_mode:
            dump_tensor(in_feat, name_template.format("output"), dump_step, dump_file)
        if Const.ALL in DumpUtil.dump_mode:
            dump_tensor(out_feat, name_template.format("input"), dump_step, dump_file)
            dump_tensor(in_feat, name_template.format("output"), dump_step, dump_file)
    elif Const.BACKWARD not in name_template and Const.BACKWARD not in DumpUtil.dump_mode:
        if 'input' in DumpUtil.dump_mode:
            dump_tensor(in_feat, name_template.format("input"), dump_step, dump_file)
        if 'output' in DumpUtil.dump_mode:
            dump_tensor(out_feat, name_template.format("output"), dump_step, dump_file)
        if Const.ALL in DumpUtil.dump_mode:
            dump_tensor(in_feat, name_template.format("input"), dump_step, dump_file)
            dump_tensor(out_feat, name_template.format("output"), dump_step, dump_file)


def dump_acc_cmp(name, in_feat, out_feat, dump_step, module):
    dump_file = DumpUtil.get_dump_path()
    dump_file = modify_dump_path(dump_file, DumpUtil.dump_switch_mode)
    _set_dump_switch4api_list(name)
    if DumpUtil.get_dump_switch():
        rank = get_tensor_rank(in_feat, out_feat)
        if DumpUtil.target_rank is not None:
            if rank != DumpUtil.target_rank:
                return
        dump_file = create_dirs_if_not_exist(rank, dump_file)
        global pkl_name
        pkl_name = dump_file
        if DumpUtil.dump_init_enable:
            DumpUtil.dump_init_enable = False
            DumpUtil.dump_data_dir = make_dump_data_dir(dump_file) \
                if DumpUtil.dump_switch_mode not in [Const.STACK, Const.ACL] and not DumpUtil.summary_only else ""
            if os.path.exists(dump_file) and not os.path.isdir(dump_file):
                check_writable(dump_file)
                os.remove(dump_file)

        name_prefix = name
        name_template = f"{name_prefix}" + "_{}"
        if DumpUtil.dump_switch_mode in [Const.ALL, Const.API_LIST]:
            dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file)
        elif DumpUtil.dump_switch_mode == Const.API_STACK:
            dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file)
            dump_stack_info(name_template, dump_file)
        elif DumpUtil.check_switch_scope(name_prefix):
            if DumpUtil.dump_switch_mode == Const.ACL:
                acl_dump(module, name, name_prefix)
            elif DumpUtil.dump_switch_mode != Const.STACK:
                dump_api_tensor(dump_step, in_feat, name_template, out_feat, dump_file)
            dump_stack_info(name_template, dump_file)


def acl_dump(module, module_name, name_prefix):
    if name_prefix in DumpUtil.backward_input:
        dump_mode_backward_acl_dump(module, module_name, DumpUtil.backward_input.get(name_prefix))
    else:
        forward_acl_dump(module, module_name)


def Op_Need_Trigger(module_name):
    if 'Tensor___getitem___' in module_name:
        return True
    return False


def forward_acl_dump(module, module_name):
    global forward_init_status
    global backward_init_status
    if not forward_init_status and not backward_init_status:
        forward_init_status = True
        torch_npu.npu.init_dump()
        torch_npu.npu.set_dump(DumpUtil.dump_config)
        torch_npu.npu.synchronize()
        if Op_Need_Trigger(module_name):
            module.forward(*module.input_args, **module.input_kwargs).cpu()
        else:
            module.forward(*module.input_args, **module.input_kwargs)
        torch_npu.npu.synchronize()
        torch_npu.npu.finalize_dump()
    del module.input_args
    del module.input_kwargs
    forward_init_status = False
    print_info_log("Dump %s op file." % module_name)


def acl_backward_dump_status(output, grad, module_name):
    if isinstance(output, torch.Tensor):
        output.backward(grad, retain_graph=True)
        return True

    for api_name in multi_output_apis:
        if api_name in module_name:
            output[0].backward(grad, retain_graph=True)
            return True
    return False


def dump_mode_backward_acl_dump(module, module_name, grad_path):
    global forward_init_status
    global backward_init_status
    module_name = module_name.replace(Const.FORWARD, Const.BACKWARD)
    if not forward_init_status and not backward_init_status:
        forward_init_status = True
        module.input_args = list(module.input_args)
        for i, data in enumerate(module.input_args):
            if isinstance(data, torch.Tensor) and data.grad_fn:
                module.input_args[i] = data.detach().requires_grad_()
        output = module.forward(*module.input_args, **module.input_kwargs)
        grad = torch.tensor(np.load(grad_path)).to("npu").requires_grad_()
        torch_npu.npu.init_dump()
        torch_npu.npu.set_dump(DumpUtil.dump_config)
        torch_npu.npu.synchronize()
        if not acl_backward_dump_status(output, grad, module_name):
            print_warn_log("The output of {} is not of tensor type and cannot be automatically derived. "
                            "you can manually construct a single API backward case for ACL dump.".format(module_name))
        torch_npu.npu.synchronize()
        torch_npu.npu.finalize_dump()
    del module.input_args
    del module.input_kwargs
    forward_init_status = False
    print_info_log("Dump %s op file." % module_name)


def acc_cmp_dump(name, **kwargs):
    dump_step = kwargs.get('dump_step', 1)
    pid = kwargs.get('pid')
    if not pid:
        return RuntimeError("Not get the specified process pid.")

    def acc_cmp_hook(module, in_feat, out_feat):
        if pid == os.getpid():
            dump_acc_cmp(name, in_feat, out_feat, dump_step, module)
        if hasattr(module, "input_args"):
            del module.input_args
        if hasattr(module, "input_kwargs"):
            del module.input_kwargs

    return acc_cmp_hook


def write_to_disk():
    if api_list:
        with open(pkl_name, 'a') as f:
            try:
                f.write('\n'.join(json.dumps(item) for item in api_list))
                f.write('\n')
            except:
                raise Exception("write to disk failed")


def get_pkl_file_path():
    return pkl_name
