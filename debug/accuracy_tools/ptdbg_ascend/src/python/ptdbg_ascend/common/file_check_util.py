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
import re

from .log import print_warn_log, print_error_log


class FileCheckConst:
    """
    Class for file check const
    """
    READ_ABLE = "read"
    WRITE_ABLE = "write"
    READ_WRITE_ABLE = "read and write"
    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FILE_VALID_PATTERN = r"^[a-zA-Z0-9_.:/-]+$"
    PKL_SUFFIX = ".pkl"
    NUMPY_SUFFIX = ".npy"
    JSON_SUFFIX = ".json"
    PT_SUFFIX = ".pt"
    CSV_SUFFIX = ".csv"
    MAX_PKL_SIZE = 1 * 1024 * 1024 * 1024
    MAX_NUMPY_SIZE = 10 * 1024 * 1024 * 1024
    MAX_JSON_SIZE = 1 * 1024 * 1024 * 1024
    MAX_PT_SIZE = 10 * 1024 * 1024 * 1024
    MAX_CSV_SIZE = 1 * 1024 * 1024 * 1024
    DIR = "dir"
    FILE = "file"
    DATA_DIR_AUTHORITY = 0o750
    DATA_FILE_AUTHORITY = 0o640


class FileCheckException(Exception):
    """
    Class for File Check Exception
    """
    NONE_ERROR = 0
    INVALID_PATH_ERROR = 1
    INVALID_FILE_TYPE_ERROR = 2
    INVALID_PARAM_ERROR = 3
    INVALID_PERMISSION_ERROR = 3

    def __init__(self, code, error_info: str = ""):
        super(FileCheckException, self).__init__()
        self.code = code
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class FileChecker:
    """
    The class for check file.

    Attributes:
        file_path: The file or dictionary path to be verified.
        path_type: file or dictionary
        ability(str): FileCheckConst.WRITE_ABLE or FileCheckConst.READ_ABLE to set file has writability or readability
        file_type(str): The correct file type for file
    """
    def __init__(self, file_path, path_type, ability=None, file_type=None):
        self.file_path = file_path
        self.path_type = self._check_path_type(path_type)
        self.ability = ability
        self.file_type = file_type

    @staticmethod
    def _check_path_type(path_type):
        if path_type not in [FileCheckConst.DIR, FileCheckConst.FILE]:
            print_error_log(f'The path_type must be {FileCheckConst.DIR} or {FileCheckConst.FILE}.')
            raise FileCheckException(FileCheckException.INVALID_PARAM_ERROR)
        return path_type

    def common_check(self):
        """
        功能：用户校验基本文件权限：软连接、文件长度、是否存在、读写权限、文件属组、文件特殊字符
        注意：文件后缀的合法性，非通用操作，可使用其他独立接口实现
        """
        check_link(self.file_path)
        check_path_length(self.file_path)
        check_path_exists(self.file_path)
        check_path_type(self.file_path, self.path_type)
        self.check_path_ability()
        check_path_owner_consistent(self.file_path)
        check_path_pattern_vaild(self.file_path)
        check_common_file_size(self.file_path)
        check_file_suffix(self.file_path, self.file_type)
        return os.path.realpath(self.file_path)

    def check_path_ability(self):
        if self.ability == FileCheckConst.WRITE_ABLE:
            check_path_writability(self.file_path)
        if self.ability == FileCheckConst.READ_ABLE:
            check_path_readability(self.file_path)
        if self.ability == FileCheckConst.READ_WRITE_ABLE:
            check_path_readability(self.file_path)
            check_path_writability(self.file_path)


class FileOpen:
    """
    The class for open file by a safe way.

    Attributes:
        file_path: The file or dictionary path to be opened.
        mode(str): The file open mode
    """
    SUPPORT_READ_MODE = ["r", "rb"]
    SUPPORT_WRITE_MODE = ["w", "wb", "a", "ab"]
    SUPPORT_READ_WRITE_MODE = ["r+", "rb+", "w+", "wb+", "a+", "ab+"]

    def __init__(self, file_path, mode):
        self.file_path = file_path
        self.mode = mode
        self._handle = None

    def __enter__(self):
        self.check_file_path()
        self._handle = open(self.file_path, self.mode)
        return self._handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle:
            self._handle.close()

    def check_file_path(self):
        support_mode = self.SUPPORT_READ_MODE + self.SUPPORT_WRITE_MODE + self.SUPPORT_READ_WRITE_MODE
        if self.mode not in support_mode:
            print_error_log("File open not support %s mode" % self.mode)
            raise FileCheckException(FileCheckException.INVALID_PARAM_ERROR)
        check_link(self.file_path)
        check_path_length(self.file_path)
        self.check_ability_and_owner()
        check_path_pattern_vaild(self.file_path)
        if os.path.exists(self.file_path):
            check_common_file_size(self.file_path)

    def check_ability_and_owner(self):
        if self.mode in self.SUPPORT_READ_MODE:
            check_path_exists(self.file_path)
            check_path_readability(self.file_path)
            check_path_owner_consistent(self.file_path)
        if self.mode in self.SUPPORT_WRITE_MODE and os.path.exists(self.file_path):
            check_path_writability(self.file_path)
            check_path_owner_consistent(self.file_path)
        if self.mode in self.SUPPORT_READ_WRITE_MODE and os.path.exists(self.file_path):
            check_path_readability(self.file_path)
            check_path_writability(self.file_path)
            check_path_owner_consistent(self.file_path)


def check_link(path):
    abs_path = os.path.abspath(path)
    if os.path.islink(abs_path):
        print_error_log('The file path {} is a soft link.'.format(path))
        raise FileCheckException(FileCheckException.INVALID_PATH_ERROR)


def check_path_length(path):
    if len(os.path.realpath(path)) > FileCheckConst.DIRECTORY_LENGTH or \
            len(os.path.basename(path)) > FileCheckConst.FILE_NAME_LENGTH:
        print_error_log('The file path length exceeds limit.')
        raise FileCheckException(FileCheckException.INVALID_PATH_ERROR)


def check_path_exists(path):
    real_path = os.path.realpath(path)
    if not os.path.exists(real_path):
        print_error_log('The file path %s does not exist.' % path)
        raise FileCheckException(FileCheckException.INVALID_PATH_ERROR)


def check_path_readability(path):
    real_path = os.path.realpath(path)
    if not os.access(real_path, os.R_OK):
        print_error_log('The file path %s is not readable.' % path)
        raise FileCheckException(FileCheckException.INVALID_PERMISSION_ERROR)


def check_path_writability(path):
    real_path = os.path.realpath(path)
    if not os.access(real_path, os.W_OK):
        print_error_log('The file path %s is not writable.' % path)
        raise FileCheckException(FileCheckException.INVALID_PERMISSION_ERROR)


def _user_interactive_confirm(message):
    while True:
        check_message = input(message + " Enter 'c' to continue or enter 'e' to exit: ")
        if check_message == "c":
            break
        elif check_message == "e":
            print_warn_log("User canceled.")
            raise FileCheckException(FileCheckException.INVALID_PATH_ERROR)
        else:
            print("Input is error, please enter 'c' or 'e'.")


def check_path_owner_consistent(path):
    real_path = os.path.realpath(path)
    file_owner = os.stat(real_path).st_uid
    if file_owner != os.getuid():
        _user_interactive_confirm('The file path %s may be insecure because is does not belong to you.'
                                  'Do you want to continue?' % path)


def check_path_pattern_vaild(path):
    if not re.match(FileCheckConst.FILE_VALID_PATTERN, os.path.realpath(path)):
        print_error_log('The file path {} contains special characters.'.format(path))
        raise FileCheckException(FileCheckException.INVALID_PATH_ERROR)


def check_file_size(file_path, max_size):
    real_path = os.path.realpath(file_path)
    file_size = os.path.getsize(real_path)
    if file_size >= max_size:
        _user_interactive_confirm(f'The size of file path {file_path} exceeds {max_size} bytes.'
                                  f'Do you want to continue?')


def check_common_file_size(file_path):
    if os.path.isfile(file_path):
        if file_path.endswith(FileCheckConst.PKL_SUFFIX):
            check_file_size(file_path, FileCheckConst.MAX_PKL_SIZE)
        if file_path.endswith(FileCheckConst.NUMPY_SUFFIX):
            check_file_size(file_path, FileCheckConst.MAX_NUMPY_SIZE)
        if file_path.endswith(FileCheckConst.JSON_SUFFIX):
            check_file_size(file_path, FileCheckConst.MAX_JSON_SIZE)
        if file_path.endswith(FileCheckConst.PT_SUFFIX):
            check_file_size(file_path, FileCheckConst.MAX_PT_SIZE)
        if file_path.endswith(FileCheckConst.CSV_SUFFIX):
            check_file_size(file_path, FileCheckConst.MAX_CSV_SIZE)


def check_file_suffix(file_path, file_suffix):
    if file_suffix:
        real_path = os.path.realpath(file_path)
        if not real_path.endswith(file_suffix):
            print_error_log(f"The {file_path} should be a {file_suffix} file!")
            raise FileCheckException(FileCheckException.INVALID_FILE_TYPE_ERROR)


def check_path_type(file_path, file_type):
    real_path = os.path.realpath(file_path)
    if file_type == FileCheckConst.FILE:
        if not os.path.isfile(real_path):
            print_error_log(f"The {file_path} should be a file!")
            raise FileCheckException(FileCheckException.INVALID_FILE_TYPE_ERROR)
    if file_type == FileCheckConst.DIR:
        if not os.path.isdir(real_path):
            print_error_log(f"The {file_path} should be a dictionary!")
            raise FileCheckException(FileCheckException.INVALID_FILE_TYPE_ERROR)


def create_directory(dir_path):
    """
    Function Description:
        creating a directory with specified permissions
    Parameter:
        dir_path: directory path
    Exception Description:
        when invalid data throw exception
    """
    dir_path = os.path.realpath(dir_path)
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, mode=FileCheckConst.DATA_DIR_AUTHORITY)
        except OSError as ex:
            print_error_log(
                'Failed to create {}.Please check the path permission or disk space .{}'.format(dir_path, str(ex)))
            raise FileCheckException(FileCheckException.INVALID_PATH_ERROR)


def change_mode(path, mode):
    if not os.path.exists(path) or os.path.islink(path):
        return
    try:
        os.chmod(path, mode)
    except PermissionError as ex:
        print_error_log('Failed to change {} authority. {}'.format(path, str(ex)))
        raise FileCheckException(FileCheckException.INVALID_PERMISSION_ERROR)

