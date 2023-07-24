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

from .utils import DumpUtil, _set_dump_switch4api_list, make_dump_data_dir

from ..common.utils import print_warn_log, Const, print_info_log, modify_dump_path
from ..dump.utils import check_writable

forward_init_status = False
backward_init_status = False

backward_threading_id = 0


class DataInfo(object):
    def __init__(self, data, save_data, summary_data, dtype, shape):
        self.data = data
        self.save_data = save_data
        self.summary_data = summary_data
        self.dtype = dtype
        self.shape = shape


def get_not_float_tensor_info(data):
    summary_data = []
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
    saved_tensor = data.contiguous().cpu().detach().numpy()
    summary_data.extend([tensor_max, tensor_min, tensor_mean])
    return DataInfo(data, saved_tensor, summary_data, str(data.dtype), tuple(data.shape))


def get_scalar_data_info(data):
    summary_data = [data, data, data]
    return DataInfo(data, data, summary_data, str(type(data)), str([]))


def get_float_tensor_info(data):
    summary_data = []
    tensor_max = torch._C._VariableFunctionsClass.max(data).cpu().detach().float().numpy().tolist()
    tensor_min = torch._C._VariableFunctionsClass.min(data).cpu().detach().float().numpy().tolist()
    tensor_mean = torch._C._VariableFunctionsClass.mean(data).cpu().detach().float().numpy().tolist()
    saved_tensor = data.contiguous().cpu().detach().numpy()
    summary_data.extend([tensor_max, tensor_min, tensor_mean])
    return DataInfo(data, saved_tensor, summary_data, str(data.dtype), tuple(data.shape))


def dump_tensor(args):
    global data_info
    args_list = []
    for x in args:
        if isinstance(x, torch.Tensor):
            if x.numel() == 0 or len(x.shape) == 0 or not x.is_floating_point():
                data_info = get_not_float_tensor_info(x)
            else:
                data_info = get_float_tensor_info(x)
            arg = {"dtype": data_info.dtype,
                   "shape": data_info.shape,
                   "type": "torch.Tensor",
                   "Max": data_info.summary_data[0],
                   "Min": data_info.summary_data[1]}
        else:
            arg = {"value": None,
                   "type": type(x)}
        args_list.append(arg)
    return args_list


def dump_api_tensor(module, name_template, out_feat, dump_file):
    api_params_dict = dict()
    api_dict = dict()
    if Const.BACKWARD in name_template and DumpUtil.dump_mode != Const.FORWARD:
        path = os.path.dirname(dump_file)
        dump_file = os.path.join(path, "dump_backward.pkl")
        api_params_dict["args"] = dump_tensor(out_feat)
    elif Const.BACKWARD not in name_template and DumpUtil.dump_mode != Const.BACKWARD:
        if module.input_args:
            args_list = dump_tensor(module.input_args)
            api_params_dict["args"] = args_list
        if module.input_kwargs:
            api_params_dict["kwargs"] = module.input_kwargs
    api_dict[name_template] = api_params_dict
    with os.fdopen(os.open(dump_file, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR),
                   "a") as f:
        json.dump(api_dict, f)
        f.write('\n')


def dump_acc_cmp(name, out_feat, module):
    dump_file = DumpUtil.get_dump_path()
    _set_dump_switch4api_list(name)

    dump_file = modify_dump_path(dump_file, DumpUtil.dump_switch_mode)

    if DumpUtil.get_dump_switch():
        if DumpUtil.dump_init_enable:
            DumpUtil.dump_init_enable = False
            DumpUtil.dump_data_dir = make_dump_data_dir(dump_file) \
                if DumpUtil.dump_switch_mode not in [Const.STACK, Const.ACL] else ""
            if os.path.exists(dump_file) and not os.path.isdir(dump_file):
                check_writable(dump_file)
                os.remove(dump_file)

        if DumpUtil.dump_switch_mode in [Const.ALL, Const.API_LIST]:
            dump_api_tensor(module, name, out_feat, dump_file)




def acc_cmp_dump(name):

    def acc_cmp_hook(module, in_feat, out_feat):
        dump_acc_cmp(name, out_feat, module)
        if hasattr(module, "input_args"):
            del module.input_args
        if hasattr(module, "input_kwargs"):
            del module.input_kwargs

    return acc_cmp_hook
