import unittest
import torch.distributed as dist
from ptdbg_ascend.hook_module.wrap_distributed import *

class TestWrapDistributed(unittest.TestCase):
    def setUp(self):
        self.hook = lambda x: x

    def test_get_distributed_ops(self):
        ops = get_distributed_ops()
        self.assertIsInstance(ops, set)

    def test_DistributedOPTemplate(self):
        op_name = 'all_reduce'
        if op_name in get_distributed_ops():
            op = DistributedOPTemplate(op_name, self.hook)
            self.assertEqual(op.op_name_, op_name)

    def test_wrap_distributed_op(self):
        op_name = 'all_reduce'
        if op_name in get_distributed_ops():
            wrapped_op = wrap_distributed_op(op_name, self.hook)
            self.assertTrue(callable(wrapped_op))

    def test_wrap_distributed_ops_and_bind(self):
        wrap_distributed_ops_and_bind(self.hook)
        for op_name in get_distributed_ops():
            self.assertTrue(hasattr(HOOKDistributedOP, "wrap_" + str(op_name)))

