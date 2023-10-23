# coding=utf-8
from api_accuracy_checker.run_ut.run_ut import generate_cpu_params, get_api_info
import unittest
import numpy as np
import os
import copy
from api_accuracy_checker.run_ut.run_ut import *
from api_accuracy_checker.common.utils import get_json_contents

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
forward_file = os.path.join(base_dir, "../resources/forward.json")
forward_content = get_json_contents(forward_file)
for api_full_name, api_info_dict in forward_content.items():
    api_full_name = api_full_name
    api_info_dict = api_info_dict
    
class TestRunUtMethods(unittest.TestCase):
    def test_exec_api(self):
        api_info = copy.deepcopy(api_info_dict)
        [api_type, api_name, _] = api_full_name.split("*")
        args, kwargs, need_grad = get_api_info(api_info, api_name)
        cpu_args, cpu_kwargs = generate_cpu_params(args, kwargs, True)
        out = exec_api(api_type, api_name, cpu_args, cpu_kwargs)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.requires_grad, True)
        self.assertEqual(out.shape, torch.Size([2, 2560, 24, 24]))
        
    def test_generate_npu_params(self):
        api_info = copy.deepcopy(api_info_dict)
        [api_type, api_name, _] = api_full_name.split("*")
        args, kwargs, need_grad = get_api_info(api_info, api_name)
        npu_args, npu_kwargs = generate_npu_params(args, kwargs, True)
        self.assertEqual(len(npu_args), 1)
        self.assertEqual(npu_args[0].dtype, torch.float16)
        self.assertEqual(npu_args[0].requires_grad, True)
        self.assertEqual(npu_args[0].shape, torch.Size([2, 2560, 24, 24]))
        self.assertEqual(npu_kwargs, {'inplace': False})
        
    def test_generate_cpu_params(self):
        api_info = copy.deepcopy(api_info_dict)
        [api_type, api_name, _] = api_full_name.split("*")
        args, kwargs, need_grad = get_api_info(api_info, api_name)
        cpu_args, cpu_kwargs = generate_cpu_params(args, kwargs, True)
        self.assertEqual(len(cpu_args), 1)
        self.assertEqual(cpu_args[0].dtype, torch.float16)
        self.assertEqual(cpu_args[0].requires_grad, True)
        self.assertEqual(cpu_args[0].shape, torch.Size([2, 2560, 24, 24]))
        self.assertEqual(cpu_kwargs, {'inplace': False})