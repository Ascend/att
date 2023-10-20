import unittest
import torch
from api_accuracy_checker.common.base_api import BaseAPIInfo

class TestBaseAPI(unittest.TestCase):
    def setUp(self):
        # Initialize the BaseAPIInfo object
        self.api = BaseAPIInfo("test_api", True, True, "/path/to/save", "forward", "backward")

    def test_analyze_element(self):
        # Test analyze_element method
        element = [1, 2, 3]
        result = self.api.analyze_element(element)
        self.assertEqual(result, {'type': 'list', 'value': [1, 2, 3]})

    def test_analyze_tensor(self):
        # Test analyze_tensor method
        tensor = torch.tensor([1, 2, 3])
        result = self.api.analyze_tensor(tensor)
        self.assertEqual(result['type'], 'torch.Tensor')
        self.assertEqual(result['dtype'], 'torch.int64')
        self.assertEqual(result['shape'], (3,))
        self.assertEqual(result['Max'], 3)
        self.assertEqual(result['Min'], 1)
        self.assertEqual(result['requires_grad'], False)

    def test_analyze_builtin(self):
        # Test analyze_builtin method
        arg = slice(1, 10, 2)
        result = self.api.analyze_builtin(arg)
        self.assertEqual(result, {'type': 'slice', 'value': [1, 10, 2]})

    def test_transfer_types(self):
        # Test transfer_types method
        data = 10
        dtype = 'int'
        result = self.api.transfer_types(data, dtype)
        self.assertEqual(result, 10)

    def test_is_builtin_class(self):
        # Test is_builtin_class method
        element = 10
        result = self.api.is_builtin_class(element)
        self.assertEqual(result, True)

    def test_analyze_device_in_kwargs(self):
        # Test analyze_device_in_kwargs method
        element = torch.device('cuda:0')
        result = self.api.analyze_device_in_kwargs(element)
        self.assertEqual(result, {'type': 'torch.device', 'value': 'cuda:0'})

    def test_analyze_dtype_in_kwargs(self):
        # Test analyze_dtype_in_kwargs method
        element = torch.float32
        result = self.api.analyze_dtype_in_kwargs(element)
        self.assertEqual(result, {'type': 'torch.dtype', 'value': 'torch.float32'})

    def test_get_tensor_extremum(self):
        # Test get_tensor_extremum method
        data = torch.tensor([1, 2, 3])
        result_max = self.api.get_tensor_extremum(data, 'max')
        result_min = self.api.get_tensor_extremum(data, 'min')
        self.assertEqual(result_max, 3)
        self.assertEqual(result_min, 1)

    def test_get_type_name(self):
        # Test get_type_name method
        name = "<class 'int'>"
        result = self.api.get_type_name(name)
        self.assertEqual(result, 'int')

if __name__ == '__main__':
    unittest.main()