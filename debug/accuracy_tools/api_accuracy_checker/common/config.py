import yaml
import os
from api_accuracy_checker.common.utils import check_file_or_directory_path

class Config:
    def __init__(self, yaml_file):
        check_file_or_directory_path(yaml_file, False)
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        self.config = {key: self.validate(key, value) for key, value in config.items()}

    def validate(self, key, value):
        validators = {
            'dump_path': str,
            'jit_compile': bool,
            'compile_option': str,
            'compare_algorithm': str,
            'real_data': bool,
            'dump_step': int,
            'error_data_path': str,
            'enable_dataloader': bool,
            'target_iter': int
        }
        if not isinstance(value, validators[key]):
            raise ValueError(f"{key} must be {validators[key].__name__} type")
        if key == 'target_iter' and value < 0:
            raise ValueError("target_iter must be greater than 0")
        return value

    def __str__(self):
        return '\n'.join(f"{key}={value}" for key, value in self.config.items())

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = self.validate(key, value)
            else:
                raise ValueError(f"Invalid key '{key}'")


cur_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
yaml_path = os.path.join(cur_path, "config.yaml")
msCheckerConfig = Config(yaml_path)