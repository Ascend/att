import os
import json
import numpy as np
from .utils import DumpUtil
from .api_info import ForwardAPIInfo, BackwardAPIInfo
import threading
import fcntl
from ..common.utils import check_file_or_directory_path

lock = threading.Lock()

def write_api_info_json(api_info):
    dump_path = DumpUtil.dump_path
    initialize_output_json()
    if isinstance(api_info, ForwardAPIInfo):
        file_path = os.path.join(dump_path, 'forward_info.json')
        stack_file_path = os.path.join(dump_path, 'stack_info.json')
        write_json(file_path, api_info.api_info_struct)
        write_json(stack_file_path, api_info.stack_info_struct, indent=4)

    elif isinstance(api_info, BackwardAPIInfo):
        file_path = os.path.join(dump_path, 'backward_info.json')
        write_json(file_path, api_info.grad_info_struct)
    else:
        raise ValueError(f"Invalid api_info type {type(api_info)}")

def write_npy(name, tensor):
    dump_path = DumpUtil.dump_path
    if not os.path.exists(dump_path):
        raise ValueError(f"Path {dump_path} does not exist")
    npy_folder = os.path.join(dump_path, 'npy')
    if not os.path.exists(npy_folder):
        os.makedirs(npy_folder, mode=0o750)
    npy_path = os.path.join(npy_folder, name)
    if os.path.exists(npy_path):
        raise ValueError(f"File {npy_path} already exists")
    np.save(npy_path, tensor)

def write_json(file_path, data, indent=None):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("{\n}")
    try:
        lock.acquire()
        with open(file_path, 'a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
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
    dump_path = DumpUtil.dump_path
    check_file_or_directory_path(dump_path)
    files = ['forward_info.json', 'backward_info.json', 'stack_info.json']
    for file in files:
        file_path = os.path.join(dump_path, file)
        if os.path.exists(file_path):
            raise ValueError(f"file {file_path} already exists, please remove it first or use a new dump path")