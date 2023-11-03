import unittest
import pytest
import ptdbg_ascend.common.utils as utils

from ptdbg_ascend.common.utils import CompareException
from ptdbg_ascend.common.file_check_util import FileCheckException


class TestUtilsMethods(unittest.TestCase):
    def test_get_api_name_from_matcher(self):
        normal_name = "Functional_relu__1_output"
        unusual_name = "Functional_norm_layer_1_output"
        error_name = "Tensor_onnx::connect_1_input"
        api_name_1 = utils.get_api_name_from_matcher(normal_name)
        api_name_2 = utils.get_api_name_from_matcher(unusual_name)
        api_name_3 = utils.get_api_name_from_matcher(error_name)
        self.assertEqual(api_name_1, "relu")
        self.assertEqual(api_name_2, "norm_layer")
        self.assertEqual(api_name_3, "")

    def test_check_file_or_directory_path_1(self):
        file = "list"
        with pytest.raises(FileCheckException) as error:
            utils.check_file_or_directory_path(file)
        self.assertEqual(error.value.code, FileCheckException.INVALID_PATH_ERROR)

    def test_check_file_or_directory_path_2(self):
        file = "/list/dir"
        with pytest.raises(FileCheckException) as error:
            utils.check_file_or_directory_path(file)
        self.assertEqual(error.value.code, FileCheckException.INVALID_PATH_ERROR)

    def test_check_file_size_1(self):
        file = "/list/dir"
        with pytest.raises(CompareException) as error:
            utils.check_file_size(file, 100)
        self.assertEqual(error.value.code, CompareException.INVALID_FILE_ERROR)

    def test_check_file_size_2(self):
        file = "../run_ut.py"
        with pytest.raises(CompareException) as error:
            utils.check_file_size(file, 0)
        self.assertEqual(error.value.code, CompareException.INVALID_FILE_ERROR)

