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


from api_accuracy_checker.dump.api_info import ForwardAPIInfo, BackwardAPIInfo
from api_accuracy_checker.dump.info_dump import write_api_info_json, initialize_output_json
from api_accuracy_checker.common.utils import print_warn_log, print_info_log, print_error_log
from api_accuracy_checker.hook_module.register_hook import initialize_hook


def set_dump_switch(switch):
    if switch == "ON":
        initialize_hook(pretest_hook)
        initialize_output_json()
    DumpUtil.set_dump_switch(switch)

class DumpUtil(object):
    dump_switch = None

    @staticmethod
    def set_dump_switch(switch):
        DumpUtil.dump_switch = switch

    @staticmethod
    def get_dump_switch():
        return DumpUtil.dump_switch == "ON"


class DumpConst:
    delimiter = '*'
    forward = 'forward'
    backward = 'backward'


def pretest_info_dump(name, out_feat, module, phase):
    if not DumpUtil.dump_switch:
        return 
    if phase == DumpConst.forward:
        api_info = ForwardAPIInfo(name, module.input_args, module.input_kwargs)
    elif phase == DumpConst.backward:
        api_info = BackwardAPIInfo(name, out_feat)
    else:
        msg = "Unexpected training phase {}.".format(phase)
        print_error_log(msg)
        raise NotImplementedError(msg)
    
    write_api_info_json(api_info)

def pretest_hook(name, phase):
    def pretest_info_dump_hook(module, in_feat, out_feat):
        pretest_info_dump(name, out_feat, module, phase)
        if hasattr(module, "input_args"):
            del module.input_args 
        if hasattr(module, "input_kwargs"):
            del module.input_kwargs 
    return pretest_info_dump_hook 
