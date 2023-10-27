import unittest
from api_accuracy_checker.dump.api_info import APIInfo, ForwardAPIInfo, BackwardAPIInfo
from api_accuracy_checker.common.config import msCheckerConfig

class TestAPIInfo(unittest.TestCase):
    def test_APIInfo(self):
        api_info = APIInfo("test_api", True, True, "save_path")
        self.assertEqual(api_info.api_name, "test_api")
        self.assertEqual(api_info.is_forward, True)
        self.assertEqual(api_info.is_save_data, True)
        self.assertEqual(api_info.save_path, "save_path")
        self.assertEqual(api_info.forward_path, "forward_real_data")
        self.assertEqual(api_info.backward_path, "backward_real_data")

    def test_ForwardAPIInfo(self):
        forward_api_info = ForwardAPIInfo("test_forward_api", [1, 2, 3], {"a": 1, "b": 2})
        self.assertEqual(forward_api_info.api_name, "test_forward_api")
        self.assertEqual(forward_api_info.is_forward, True)
        self.assertEqual(forward_api_info.is_save_data, msCheckerConfig.real_data)
        self.assertEqual(forward_api_info.save_path, msCheckerConfig.dump_path)
        self.assertEqual(forward_api_info.api_info_struct, {"test_forward_api": {"args": [{'type': 'int', 'value': 1},{'type': 'int', 'value': 2},{'type': 'int', 'value': 3},], "kwargs": {'a': {'type': 'int', 'value': 1}, 'b': {'type': 'int', 'value': 2}}}})

    def test_BackwardAPIInfo(self):
        backward_api_info = BackwardAPIInfo("test_backward_api", [1, 2, 3])
        self.assertEqual(backward_api_info.api_name, "test_backward_api")
        self.assertEqual(backward_api_info.is_forward, False)
        self.assertEqual(backward_api_info.is_save_data, msCheckerConfig.real_data)
        self.assertEqual(backward_api_info.save_path, msCheckerConfig.dump_path)
        self.assertEqual(backward_api_info.grad_info_struct, {"test_backward_api": [{'type': 'int', 'value': 1},{'type': 'int', 'value': 2},{'type': 'int', 'value': 3}]})
