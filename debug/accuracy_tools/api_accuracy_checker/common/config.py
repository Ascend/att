import yaml
import os

class Config:
    def __init__(self, yaml_file):
        if not os.path.exists(yaml_file):
            raise ValueError(f"File {yaml_file} does not exist")
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        self.dump_path = config['dump_path']
        self.jit_compile = config['jit_compile']
        self.compile_option = config['compile_option']
        self.compare_algorithm = config['compare_algorithm']
        self.real_data = config['real_data']
        self.dump_step = config['dump_step']

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
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid key '{key}'")

msCheckerConfig = Config('./config.yaml')