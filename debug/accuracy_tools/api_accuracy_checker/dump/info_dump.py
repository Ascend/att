import fcntl
import json
import os
import threading
import numpy as np

from .api_info import ForwardAPIInfo, BackwardAPIInfo
from ..common.utils import check_file_or_directory_path
from ..common.config import msCheckerConfig

lock = threading.Lock()

def write_api_info_json(api_info):
    dump_path = msCheckerConfig.dump_path
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
    dump_path = msCheckerConfig.dump_path
    check_file_or_directory_path(dump_path,True)
    
    files_in_dir = os.listdir(dump_path)
    pattern = re.compile(r"(forward|backward|stack)_info_[0-9].json")
    match = re.search(pattern, ''.join(files_in_dir))
    if match:
        match_file = match.group()
        file_path = os.path.join(DumpUtil.dump_path, match_file)
        raise ValueError(f"file {file_path} already exists, please remove it first or use a new dump path")