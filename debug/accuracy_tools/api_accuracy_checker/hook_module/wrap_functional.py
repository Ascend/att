#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2023-2023. Huawei Technologies Co., Ltd. All rights reserved.
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

import os

import torch
import yaml

from api_accuracy_checker.hook_module.hook_module import HOOKModule
from api_accuracy_checker.common.utils import torch_device_guard
from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.hook_module.utils import WrapFunctionalOps
from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileOpen

for f in dir(torch.nn.functional):
    locals().update({f: getattr(torch.nn.functional, f)})


def get_functional_ops():
    global WrapFunctionalOps
    _all_functional_ops = dir(torch.nn.functional)
    if msCheckerConfig.white_list:
        return set(WrapFunctionalOps) & set(_all_functional_ops) & set(msCheckerConfig.white_list)
    else:
        return set(WrapFunctionalOps) & set(_all_functional_ops) 


class HOOKFunctionalOP(object):
    pass


class FunctionalOPTemplate(HOOKModule):
    def __init__(self, op_name, hook, need_hook=True):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Functional*" + str(op_name) + "*"
        if need_hook:
            super().__init__(hook)

    @torch_device_guard
    def forward(self, *args, **kwargs):
        return eval(self.op_name_)(*args, **kwargs)


def wrap_functional_op(op_name, hook):
    def functional_op_template(*args, **kwargs):
        return FunctionalOPTemplate(op_name, hook)(*args, **kwargs)

    return functional_op_template


def wrap_functional_ops_and_bind(hook):
    _functional_ops = get_functional_ops()
    for op_name in _functional_ops:
        setattr(HOOKFunctionalOP, "wrap_" + op_name, wrap_functional_op(op_name, hook))
