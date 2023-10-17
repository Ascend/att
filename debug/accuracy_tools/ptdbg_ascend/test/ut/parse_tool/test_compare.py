import unittest
import numpy as np
from ptdbg_ascend.parse_tool.lib.compare import Compare

class TestCompare(unittest.TestCase):
    def setUp(self):
        self.compare = Compare()

    def test_call_msaccucmp(self):
        result = self.compare.call_msaccucmp
        self.assertIsNotNone(result)

    def test_npu_vs_npu_compare(self):
        my_dump_path = 'path_to_my_dump'
        golden_dump_path = 'path_to_golden_dump'
        result_dir = 'path_to_result_dir'
        self.compare.npu_vs_npu_compare(my_dump_path, golden_dump_path, result_dir)

    def test_compare_vector(self):
        my_dump_path = 'path_to_my_dump'
        golden_dump_path = 'path_to_golden_dump'
        result_dir = 'path_to_result_dir'
        result = self.compare.compare_vector(my_dump_path, golden_dump_path, result_dir)
        self.assertIsNotNone(result)

    def test_convert_dump_to_npy(self):
        dump_file = 'path_to_dump_file'
        data_format = 'data_format'
        output = './'
        self.compare.convert_dump_to_npy(dump_file, data_format, output)

    def test_convert(self):
        dump_file = 'path_to_dump_file'
        data_format = 'data_format'
        output = 'path_to_output'
        result = self.compare.convert(dump_file, data_format, output)
        self.assertIsNotNone(result)
