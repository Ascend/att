import unittest
import torch
import os
from ptdbg_ascend.overflow_check import info_dump

class TestInfoDump(unittest.TestCase):

    def setUp(self):
        self.tensor = torch.tensor([1.0, 2.0, 3.0])
        self.file_path = './test.npy'
        self.api_info = info_dump.APIInfo('test_api', True)
        self.forward_api_info = info_dump.ForwardAPIInfo('test_api', True, (1, 2, 3), {'a': 1, 'b': 2})
        self.backward_api_info = info_dump.BackwardAPIInfo('test_api', (1, 2, 3))
        self.dump_path = './'
        self.json_file_path = os.path.join(self.dump_path, 'test.json')

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_write_npy(self):
        npy_path = info_dump.write_npy(self.file_path, self.tensor)
        self.assertTrue(os.path.exists(npy_path))

    def test_APIInfo_init(self):
        self.assertEqual(self.api_info.api_name, 'test_api')
        self.assertEqual(self.api_info.is_forward, True)

    def test_analyze_element(self):
        result = self.api_info.analyze_element(self.tensor)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['type'], 'torch.Tensor')

    def test_analyze_tensor(self):
        result = self.api_info.analyze_tensor(self.tensor, False)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['type'], 'torch.Tensor')

    def test_analyze_builtin(self):
        result = self.api_info.analyze_builtin(5)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['type'], 'int')

    def test_transfer_types(self):
        result = self.api_info.transfer_types(5, 'int')
        self.assertIsInstance(result, int)
        self.assertEqual(result, 5)

    def test_is_builtin_class(self):
        result = self.api_info.is_builtin_class(5)
        self.assertTrue(result)

    def test_analyze_device_in_kwargs(self):
        result = self.api_info.analyze_device_in_kwargs('cpu')
        self.assertIsInstance(result, dict)
        self.assertEqual(result['type'], 'torch.device')

    def test_analyze_dtype_in_kwargs(self):
        result = self.api_info.analyze_dtype_in_kwargs(torch.float32)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['type'], 'torch.dtype')

    def test_get_tensor_extremum(self):
        result = self.api_info.get_tensor_extremum(self.tensor, 'max')
        self.assertEqual(result, 3.0)

    def test_get_type_name(self):
        result = self.api_info.get_type_name(str(type(self.tensor)))
        self.assertEqual(result, 'torch.Tensor')

    def test_ForwardAPIInfo_init(self):
        self.assertEqual(self.forward_api_info.api_name, 'test_api')
        self.assertEqual(self.forward_api_info.is_forward, True)

    def test_BackwardAPIInfo_init(self):
        self.assertEqual(self.backward_api_info.api_name, 'test_api')
        self.assertEqual(self.backward_api_info.is_forward, False)

    def test_write_api_info_json(self):
        info_dump.write_api_info_json(self.forward_api_info)
        self.assertTrue(os.path.exists(os.path.join(self.dump_path, f'forward_info_{self.forward_api_info.rank}.json')))
        if os.path.exists(os.path.join(self.dump_path, f'forward_info_{self.forward_api_info.rank}.json')):
            os.remove(os.path.join(self.dump_path, f'forward_info_{self.forward_api_info.rank}.json'))

    def test_write_json(self):
        info_dump.write_json(self.json_file_path, {'test': 'data'})
        self.assertTrue(os.path.exists(self.json_file_path))
        if os.path.exists(self.json_file_path):
            os.remove(self.json_file_path)

    def test_initialize_output_json(self):
        try:
            info_dump.initialize_output_json()
        except ValueError as e:
            self.assertTrue(str(e).startswith('file'))

