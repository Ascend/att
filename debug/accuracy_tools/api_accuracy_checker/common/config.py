import yaml
import os
from api_accuracy_checker.common.utils import check_file_or_directory_path
from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileOpen

class Config:
    def __init__(self, yaml_file):
        check_file_or_directory_path(yaml_file, False)
        with FileOpen(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        self.config = {key: self.validate(key, value) for key, value in config.items()}

    def validate(self, key, value):
        validators = {
            'dump_path': str,
            'jit_compile': bool,
            'real_data': bool,
            'dump_step': int,
            'error_data_path': str,
            'enable_dataloader': bool,
            'target_iter': int,
            'precision': int
        }
        if not isinstance(value, validators.get(key)):
            raise ValueError(f"{key} must be {validators[key].__name__} type")
        if key == 'target_iter' and value < 0:
            raise ValueError("target_iter must be greater than 0")
        if key == 'precision' and value < 0:
            raise ValueError("precision must be greater than 0")
        return value

    def __getattr__(self, item):
        return self.config[item]

    def __str__(self):
        return '\n'.join(f"{key}={value}" for key, value in self.config.items())

    def update_config(self, dump_path, real_data=False, enable_dataloader=False, target_iter=1):
        args = {
            "dump_path": dump_path,
            "real_data": real_data,
            "enable_dataloader": enable_dataloader,
            "target_iter": target_iter
        }
        for key, value in args.items():
            if key in self.config:
                self.config[key] = self.validate(key, value)
            else:
                raise ValueError(f"Invalid key '{key}'")


cur_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
yaml_path = os.path.join(cur_path, "config.yaml")
msCheckerConfig = Config(yaml_path)