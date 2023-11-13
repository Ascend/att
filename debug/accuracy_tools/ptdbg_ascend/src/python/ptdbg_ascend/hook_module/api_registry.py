#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2022-2023. Huawei Technologies Co., Ltd. All rights reserved.
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

import torch
import torch.distributed as dist
from . import wrap_torch, wrap_functional, wrap_tensor, wrap_vf, wrap_distributed, wrap_aten
from .wrap_torch import get_torch_ops
from .wrap_functional import get_functional_ops
from .wrap_tensor import get_tensor_ops
from .wrap_vf import get_vf_ops
from .wrap_distributed import get_distributed_ops
from .wrap_aten import get_aten_ops
from ..common.utils import torch_without_guard_version
torch_version_above_2 = torch.__version__.split('+')[0] > '2.0'

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False
    from . import wrap_npu_custom
    from .wrap_npu_custom import WrapNpuOps


class ApiRegistry:
    def __init__(self):
        self.tensor_ori_attr = {}
        self.torch_ori_attr = {}
        self.functional_ori_attr = {}
        self.distributed_ori_attr = {}
        self.npu_distributed_ori_attr = {}
        self.vf_ori_attr = {}
        self.aten_ori_attr = {}
        self.torch_npu_ori_attr = {}

        self.tensor_hook_attr = {}
        self.torch_hook_attr = {}
        self.functional_hook_attr = {}
        self.distributed_hook_attr = {}
        self.npu_distributed_hook_attr = {}
        self.vf_hook_attr = {}
        self.aten_hook_attr = {}
        self.torch_npu_hook_attr = {}

    @staticmethod
    def store_ori_attr(ori_api_group, api_list, api_ori_attr):
        for api in api_list:
            api_ori_attr[api] = getattr(ori_api_group, api)

    @staticmethod
    def set_api_attr(api_group, attr_dict):
        for api, api_attr in attr_dict.items():
            setattr(api_group, api, api_attr)

    def api_modularity(self):
        self.set_api_attr(torch.Tensor, self.tensor_hook_attr)
        self.set_api_attr(torch, self.torch_hook_attr)
        self.set_api_attr(torch.nn.functional, self.functional_hook_attr)
        self.set_api_attr(dist, self.distributed_hook_attr)
        self.set_api_attr(dist.distributed_c10d, self.distributed_hook_attr)
        if not is_gpu and not torch_without_guard_version:
            self.set_api_attr(torch_npu.distributed, self.npu_distributed_hook_attr)
            self.set_api_attr(torch_npu.distributed.distributed_c10d, self.npu_distributed_hook_attr)
        if torch_version_above_2:
            self.set_api_attr(torch.ops.aten, self.aten_hook_attr)
        self.set_api_attr(torch._VF, self.vf_hook_attr)
        if not is_gpu:
            self.set_api_attr(torch_npu, self.torch_npu_hook_attr)

    def api_originality(self):
        self.set_api_attr(torch.Tensor, self.tensor_ori_attr)
        self.set_api_attr(torch, self.torch_ori_attr)
        self.set_api_attr(torch.nn.functional, self.functional_ori_attr)
        self.set_api_attr(dist, self.distributed_ori_attr)
        self.set_api_attr(dist.distributed_c10d, self.distributed_ori_attr)
        if not is_gpu and not torch_without_guard_version:
            self.set_api_attr(torch_npu.distributed, self.npu_distributed_ori_attr)
            self.set_api_attr(torch_npu.distributed.distributed_c10d, self.npu_distributed_ori_attr)
        if torch_version_above_2:
            self.set_api_attr(torch.ops.aten, self.aten_ori_attr)
        self.set_api_attr(torch._VF, self.vf_ori_attr)
        if not is_gpu:
            self.set_api_attr(torch_npu, self.torch_npu_ori_attr)

    def initialize_hook(self, hook):
        self.store_ori_attr(torch.Tensor, get_tensor_ops(), self.tensor_ori_attr)
        wrap_tensor.wrap_tensor_ops_and_bind(hook)
        for attr_name in dir(wrap_tensor.HOOKTensor):
            if attr_name.startswith("wrap_"):
                self.tensor_hook_attr[attr_name[5:]] = getattr(wrap_tensor.HOOKTensor, attr_name)

        self.store_ori_attr(torch, get_torch_ops(), self.torch_ori_attr)
        wrap_torch.wrap_torch_ops_and_bind(hook)
        for attr_name in dir(wrap_torch.HOOKTorchOP):
            if attr_name.startswith("wrap_"):
                self.torch_hook_attr[attr_name[5:]] = getattr(wrap_torch.HOOKTorchOP, attr_name)

        self.store_ori_attr(torch.nn.functional, get_functional_ops(), self.functional_ori_attr)
        wrap_functional.wrap_functional_ops_and_bind(hook)
        for attr_name in dir(wrap_functional.HOOKFunctionalOP):
            if attr_name.startswith("wrap_"):
                self.functional_hook_attr[attr_name[5:]] = getattr(wrap_functional.HOOKFunctionalOP, attr_name)

        self.store_ori_attr(dist, get_distributed_ops(), self.distributed_ori_attr)
        wrap_distributed.wrap_distributed_ops_and_bind(hook)
        for attr_name in dir(wrap_distributed.HOOKDistributedOP):
            if attr_name.startswith("wrap_"):
                self.distributed_hook_attr[attr_name[5:]] = getattr(wrap_distributed.HOOKDistributedOP, attr_name)
                if not is_gpu and not torch_without_guard_version:
                    self.store_ori_attr(torch_npu.distributed, get_distributed_ops(), self.npu_distributed_ori_attr)
                    self.npu_distributed_hook_attr[attr_name[5:]] = getattr(wrap_distributed.HOOKDistributedOP,
                                                                            attr_name)

        if torch_version_above_2:
            self.store_ori_attr(torch.ops.aten, get_aten_ops(), self.aten_ori_attr)
            wrap_aten.wrap_aten_ops_and_bind(hook)
            for attr_name in dir(wrap_aten.HOOKAtenOP):
                if attr_name.startswith("wrap_"):
                    self.aten_hook_attr[attr_name[5:]] = getattr(wrap_aten.HOOKAtenOP, attr_name)

        self.store_ori_attr(torch._VF, get_vf_ops(), self.vf_ori_attr)
        wrap_vf.wrap_vf_ops_and_bind(hook)
        for attr_name in dir(wrap_vf.HOOKVfOP):
            if attr_name.startswith("wrap_"):
                self.vf_hook_attr[attr_name[5:]] = getattr(wrap_vf.HOOKVfOP, attr_name)

        if not is_gpu:
            self.store_ori_attr(torch_npu, WrapNpuOps, self.torch_npu_ori_attr)
            wrap_npu_custom.wrap_npu_ops_and_bind(hook)
            for attr_name in dir(wrap_npu_custom.HOOKNpuOP):
                if attr_name.startswith("wrap_"):
                    self.torch_npu_hook_attr[attr_name[5:]] = getattr(wrap_npu_custom.HOOKNpuOP, attr_name)


api_register = ApiRegistry()
