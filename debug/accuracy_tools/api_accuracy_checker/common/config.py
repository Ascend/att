import yaml
import os
from api_accuracy_checker.common.utils import check_file_or_directory_path

class Config:
    def __init__(self, yaml_file):
        check_file_or_directory_path(yaml_file, False)
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        self.dump_path = self.validate_dump_path(config['dump_path'])
        self.jit_compile = self.validate_jit_compile(config['jit_compile'])
        self.compile_option = self.validate_compile_option(config['compile_option'])
        self.compare_algorithm = self.validate_compare_algorithm(config['compare_algorithm'])
        self.real_data = self.validate_real_data(config['real_data'])
        self.dump_step = self.validate_dump_step(config['dump_step'])

    def validate_dump_path(self, dump_path):
        if not isinstance(dump_path, str):
            raise ValueError("dump_path mast be string type")
        return dump_path

    def validate_jit_compile(self, jit_compile):
        if not isinstance(jit_compile, bool):
            raise ValueError("jit_compile mast be bool type")
        return jit_compile

    def validate_compile_option(self, compile_option):
        if not isinstance(compile_option, str):
            raise ValueError("compile_option mast be string type")
        return compile_option

    def validate_compare_algorithm(self, compare_algorithm):
        if not isinstance(compare_algorithm, str):
            raise ValueError("compare_algorithm mast be string type")
        return compare_algorithm

    def validate_real_data(self, real_data):
        if not isinstance(real_data, bool):
            raise ValueError("real_data mast be bool type")
        return real_data

    def validate_dump_step(self, dump_step):
        if not isinstance(dump_step, int):
            raise ValueError("dump_step mast be int type")
        return dump_step


    def __str__(self):
        return (
            f"dump_path={self.dump_path}\n"
            f"jit_compile={self.jit_compile}\n"
            f"compile_option={self.compile_option}\n"
            f"compare_algorithm={self.compare_algorithm}\n"
            f"real_data={self.real_data}\n"
            f"dump_step={self.dump_step}\n"
        )

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key == 'dump_path':
                    self.validate_dump_path(value)
                elif key == 'jit_compile':
                    self.validate_jit_compile(value)
                elif key == 'compile_option':
                    self.validate_compile_option(value)
                elif key == 'compare_algorithm':
                    self.validate_compare_algorithm(value)
                elif key == 'real_data':
                    self.validate_real_data(value)
                elif key == 'dump_step':
                    self.validate_dump_step(value)
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid key '{key}'")



cur_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
yaml_path = os.path.join(cur_path, "config.yaml")
msCheckerConfig = Config(yaml_path)