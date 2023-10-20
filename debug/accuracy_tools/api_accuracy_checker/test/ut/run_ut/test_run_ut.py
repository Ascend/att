import unittest
from api_accuracy_checker.run_ut.run_ut import run_ut, generate_npu_params, generate_cpu_params, exec_api, get_api_info, run_backward, UtDataInfo
import torch

class TestRunUt(unittest.TestCase):

    def setUp(self):
        self.api_type = "Functional"
        self.api_name = "relu"
        self.args = (torch.randn(3, 3),)
        self.kwargs = {}

    def test_exec_api(self):
        out = exec_api(self.api_type, self.api_name, self.args, self.kwargs)
        self.assertIsInstance(out, torch.Tensor)

    def test_generate_npu_params(self):
        npu_args, npu_kwargs = generate_npu_params(self.args, self.kwargs, True)
        self.assertIsInstance(npu_args[0], torch.Tensor)
        self.assertEqual(npu_args[0].device.type, "npu")

    def test_generate_cpu_params(self):
        cpu_args, cpu_kwargs = generate_cpu_params(self.args, self.kwargs, True)
        self.assertIsInstance(cpu_args[0], torch.Tensor)
        self.assertEqual(cpu_args[0].device.type, "cpu")

    def test_get_api_info(self):
        api_info_dict = {"args": [1.0], "kwargs": {"out": None}}
        api_name = "abs"
        args, kwargs, need_grad = get_api_info(api_info_dict, api_name)
        self.assertEqual(args, (1.0,))
        self.assertEqual(kwargs, {"out": None})
        self.assertFalse(need_grad)

    def test_UtDataInfo(self):
        data_info = UtDataInfo(None, None, None, None, None, None)
        self.assertIsNone(data_info.bench_grad_out)
        self.assertIsNone(data_info.npu_grad_out)
        self.assertIsNone(data_info.npu_out)
        self.assertIsNone(data_info.bench_out)
        self.assertIsNone(data_info.grad_in)
        self.assertIsNone(data_info.in_fwd_data_list)

if __name__ == '__main__':
    unittest.main()