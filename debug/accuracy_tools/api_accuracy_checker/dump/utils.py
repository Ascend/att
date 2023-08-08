import os
import numpy as np


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o750)
    return path

def write_npy(file_path, tensor):
    if os.path.exists(file_path):
        raise ValueError(f"File {file_path} already exists")
    np.save(file_path, tensor)
    full_path = os.path.abspath(file_path)
    return full_path
