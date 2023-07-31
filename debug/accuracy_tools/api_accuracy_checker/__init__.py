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
# ==============================================================================
"""

from .dump.utils import set_dump_switch
from .dump.info_dump import initialize_output_json 
from .hook_module.register_hook import register_hook
from .common.utils import seed_all
from .common.version import __version__
seed_all()
# 目前，以下两行代码在run UT时需要注释掉。不知道怎么规避比较好？
register_hook() 
initialize_output_json()
__all__ = ["set_dump_switch"]
