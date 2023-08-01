import os
import shutil
import sys
from pathlib import Path
import numpy as np
from ..common.utils import print_error_log, CompareException, DumpException, Const, get_time, print_info_log, \
    check_mode_valid, get_api_name_from_matcher

from ..common.version import __version__

dump_count = 0
range_begin_flag, range_end_flag = False, False

class DumpConst:
    delimiter = '*'
    forward = 'forward'
    backward = 'backward' 

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o750)
    return path

def write_npy(file_path, tensor):
    if os.path.exists(file_path):
        raise ValueError(f"File {file_path} already exists")
    np.save(file_path, tensor)
    full_path = os.path.abspath(file_path+'.npy')
    return full_path

def check_path(dump_path):
    if not os.path.exists(dump_path):
        raise ValueError(f"Path {dump_path} does not exist")
    else:
        current_permission = oct(os.stat(dump_path).st_mode)[-3:]
        if current_permission != '750':
            raise ValueError(f"Path {dump_path} does not have the required 0o750 permissions")

class DumpUtil(object):
    save_real_data = False
    dump_path = './random_data_jsons'
    dump_switch = None

    @staticmethod
    def set_dump_path(save_path):
        DumpUtil.dump_path = save_path
        DumpUtil.dump_init_enable = True

    @staticmethod
    def set_dump_switch(switch):
        DumpUtil.dump_switch = switch

    @staticmethod
    def get_dump_path():
        if DumpUtil.dump_path:
            return DumpUtil.dump_path

    @staticmethod
    def get_dump_switch():
        return DumpUtil.dump_switch == "ON"