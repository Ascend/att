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

import functools
import os

import torch

from . import wrap_torch, wrap_functional, wrap_tensor
from .hook_module import HOOKModule
from ..common.utils import check_file_or_directory_path, print_error_log, CompareException, Const, \
    print_info_log, print_warn_log, get_process_rank
from ..dump.utils import make_dump_dirs
from ..dump.dump import acc_cmp_dump

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False

make_dir_flag = True


def initialize_hook(hook):
    wrap_tensor.wrap_tensor_ops_and_bind(hook)
    for attr_name in dir(wrap_tensor.HOOKTensor):
        if attr_name.startswith("wrap_"):
            setattr(torch.Tensor, attr_name[5:], getattr(wrap_tensor.HOOKTensor, attr_name))

    wrap_torch.wrap_torch_ops_and_bind(hook)
    for attr_name in dir(wrap_torch.HOOKTorchOP):
        if attr_name.startswith("wrap_"):
            setattr(torch, attr_name[5:], getattr(wrap_torch.HOOKTorchOP, attr_name))

    wrap_functional.wrap_functional_ops_and_bind(hook)
    for attr_name in dir(wrap_functional.HOOKFunctionalOP):
        if attr_name.startswith("wrap_"):
            setattr(torch.nn.functional, attr_name[5:], getattr(wrap_functional.HOOKFunctionalOP, attr_name))


def register_hook():
    global make_dir_flag
    if make_dir_flag:
        make_dump_dirs(0)
        make_dir_flag = False
    initialize_hook(acc_cmp_dump)
