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

import os


class Const:

    MS_ACCU_CMP_PATH = '/usr/local/Ascend/ascend-toolkit/latest/tools/operator_cmp/compare/msaccucmp.py'
    ROOT_DIR = ""
    LOG_LEVEL = "NOTSET"
    DATA_ROOT_DIR = os.path.join(ROOT_DIR, 'parse_data')
    DUMP_CONVERT_DIR = os.path.join(DATA_ROOT_DIR, 'dump_convert')
    COMPARE_DIR = os.path.join(DATA_ROOT_DIR, 'compare_result')
    OFFLINE_DUMP_CONVERT_PATTERN = \
        r"^([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\.([0-9]+)(\.[0-9]+)?\.([0-9]{1,255})" \
        r"\.([a-z]+)\.([0-9]{1,255})(\.[x0-9]+)?\.npy$"
    NUMPY_PATTERN = r".*\.npy$"
    NPY_SUFFIX = ".npy"
    PKL_SUFFIX = ".pkl"
    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    ONE_GB = 1 * 1024 * 1024 * 1024
    TEN_GB = 10 * 1024 * 1024 * 1024
    HEADER = r"""    ____                     
       / __ \____ ______________ 
      / /_/ / __ `/ ___/ ___/ _ \
     / ____/ /_/ / /  (__  )  __/
    /_/    \__,_/_/  /____/\___/ 
    
    """
