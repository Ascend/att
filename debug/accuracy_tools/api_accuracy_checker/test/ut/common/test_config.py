import unittest
import os
from api_accuracy_checker.common.config import Config

class TestConfig(unittest.TestCase):
    def setUp(self):
        cur_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        yaml_path = os.path.join(cur_path, "config.yaml")
        self.yaml_file = yaml_path
        self.config = Config(self.yaml_file)

    def test_validate(self):
        # Test valid values
        self.assertEqual(self.config.validate('dump_path', '/path/to/dump'), '/path/to/dump')
        self.assertEqual(self.config.validate('jit_compile', True), True)
        self.assertEqual(self.config.validate('compile_option', '-O3'), '-O3')

        with self.assertRaises(ValueError):
            self.config.validate('dump_path', 123)
        with self.assertRaises(ValueError):
            self.config.validate('jit_compile', 'True')

    def test_update_config(self):
        # Test updating existing keys
        self.config.update_config(dump_path='/new/path/to/dump', jit_compile=False)
        self.assertEqual(self.config.dump_path, '/new/path/to/dump')
        self.assertEqual(self.config.jit_compile, False)

        with self.assertRaises(ValueError):
            self.config.update_config(invalid_key='value')
