import unittest
from api_accuracy_checker.run_ut.run_ut import UtDataInfo

class TestRunUt(unittest.TestCase):

    def test_UtDataInfo(self):
        data_info = UtDataInfo(None, None, None, None, None, None)
        self.assertIsNone(data_info.bench_grad_out)
        self.assertIsNone(data_info.npu_grad_out)
        self.assertIsNone(data_info.npu_out)
        self.assertIsNone(data_info.bench_out)
        self.assertIsNone(data_info.grad_in)
        self.assertIsNone(data_info.in_fwd_data_list)
