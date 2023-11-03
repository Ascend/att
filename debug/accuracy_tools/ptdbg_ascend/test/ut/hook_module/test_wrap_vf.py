import unittest
import torch
from ptdbg_ascend.hook_module import wrap_vf

class TestWrapVf(unittest.TestCase):
    def setUp(self):
        self.hook = lambda x: x

    def test_get_vf_ops(self):
        ops = wrap_vf.get_vf_ops()
        self.assertIsInstance(ops, list)