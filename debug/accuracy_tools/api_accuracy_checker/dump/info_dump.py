import fcntl
import json
import os
import threading
import numpy as np

from .api_info import ForwardAPIInfo, BackwardAPIInfo
from ..common.utils import check_file_or_directory_path, initialize_save_path
from ..common.config import msCheckerConfig

lock = threading.Lock()

def write_api_info_json(api_info):
    dump_path = msCheckerConfig.dump_path
    rank = api_info.rank
    if isinstance(api_info, ForwardAPIInfo):
        file_path = os.path.join(dump_path, f'forward_info_{rank}.json')
        stack_file_path = os.path.join(dump_path, f'stack_info_{rank}.json')
        write_json(file_path, api_info.api_info_struct)
        write_json(stack_file_path, api_info.stack_info_struct, indent=4)

    elif isinstance(api_info, BackwardAPIInfo):
        file_path = os.path.join(dump_path, f'backward_info_{rank}.json')
        write_json(file_path, api_info.grad_info_struct)
    else:
        raise ValueError(f"Invalid api_info type {type(api_info)}")

def write_json(file_path, data, indent=None):
    check_file_or_directory_path(os.path.dirname(file_path),True)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("{\n}")
    lock.acquire()
    with open(file_path, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 1, os.SEEK_SET)
            f.truncate()
            if f.tell() > 3:
                f.seek(f.tell() - 1, os.SEEK_SET)
                f.truncate()
                f.write(',\n')
            f.write(json.dumps(data, indent=indent)[1:-1] + '\n}')
        except Exception as e:
            raise ValueError(f"Json save failed:{e}")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            lock.release()


def initialize_output_json():
    dump_path = os.path.realpath(msCheckerConfig.dump_path)
    check_file_or_directory_path(dump_path, True)
    files = ['forward_info.json', 'backward_info.json', 'stack_info.json']
    if msCheckerConfig.real_data:
        initialize_save_path(dump_path, 'forward_real_data')
        initialize_save_path(dump_path, 'backward_real_data')
    for file in files:
        file_path = os.path.join(dump_path, file)
        if os.path.exists(file_path):
            raise ValueError(f"file {file_path} already exists, please remove it first or use a new dump path")
