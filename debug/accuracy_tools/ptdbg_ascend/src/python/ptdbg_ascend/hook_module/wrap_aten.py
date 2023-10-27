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

from .hook_module import HOOKModule
from ..common.utils import torch_device_guard
from ..common.file_check_util import FileOpen


cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with FileOpen(yaml_path, 'r') as f:
    WrapAtenOps = yaml.safe_load(f).get('aten')


aten_func = {}
for f in dir(torch.ops.aten):
    aten_func[f] = getattr(torch.ops.aten, f)


def get_aten_ops():
    global WrapAtenOps
    _all_aten_ops = dir(torch.ops.aten)
    return set(WrapAtenOps) & set(_all_aten_ops)


class HOOKAtenOP(object):
    pass


class AtenOPTemplate(HOOKModule):
    def __init__(self, op_name, hook):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Aten_" + str(op_name) + "_"
        super().__init__(hook)

    @torch_device_guard
    def forward(self, *args, **kwargs):
        return aten_func.get(self.op_name_)(*args, **kwargs)


def wrap_aten_op(op_name, hook):
    def aten_op_template(*args, **kwargs):
        return AtenOPTemplate(op_name, hook)(*args, **kwargs)

    return aten_op_template


def wrap_aten_ops_and_bind(hook):
    _aten_ops = get_aten_ops()
    for op_name in _aten_ops:
        setattr(HOOKAtenOP, "wrap_" + str(op_name), wrap_aten_op(op_name, hook))
