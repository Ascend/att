# 定义比对算法及比对标准

import torch 
import numpy as np 
from api_accuracy_checker.compare.compare_utils import CompareConst 
from api_accuracy_checker.common.utils import print_warn_log, Const

def compare_torch_tensor(cpu_output, npu_output, compare_alg):
    if cpu_output.dtype == torch.bool:
        return compare_bool_tensor(cpu_output, npu_output)
    return compare_alg(cpu_output, npu_output)


def compare_bool_tensor(cpu_output, npu_output):
    error_rate = CompareConst.NAN
    cpu_shape = cpu_output.shape 
    npu_shape = npu_output.shape 
    if cpu_shape != npu_shape:
        return error_rate, False
    npu_data = npu_output.cpu().detach().numpy() 
    bench_data = cpu_output.detach().numpy()
    data_size = bench_data.size 
    error_nums = (bench_data != npu_data).sum()
    error_rate = float(error_nums / data_size)
    return error_rate, error_rate < 0.001 


def get_max_rel_err(n_value, b_value):
    if not isinstance(n_value, np.ndarray) or not isinstance(b_value, np.ndarray):
        print_warn_log("Max rel err only support numpy array!")
        raise ValueError("Max rel err only support numpy array!")
    if n_value.dtype != b_value.dtype:
        return CompareConst.NA, False
    if n_value.dtype in Const.FLOAT_TYPE:
        rel_err = np.abs((n_value - b_value) / (b_value + np.finfo(b_value.dtype).eps)).max()
        return rel_err, rel_err < 0.001
    if np.all(n_value == b_value):
        return 0, True 
    return 1, False 


def cosine_standard(compare_result):
    bool_result = np.array(compare_result) > 0.99
    return np.all(bool_result), bool_result


def cosine_sim(cpu_output, npu_output):
    n_value = npu_output.cpu().detach().numpy().reshape(-1)
    b_value = cpu_output.detach().numpy().reshape(-1)
    cos = CompareConst.NA 
    np.seterr(divide="ignore", invalid="ignore")
    if len(n_value) == 1:
        print_warn_log("All the data in npu dump data is scalar. Compare by relative error.")
        return get_max_rel_err(n_value, b_value)
    num = n_value.dot(b_value)
    a_norm = np.linalg.norm(n_value)
    b_norm = np.linalg.norm(b_value)
    if a_norm <= np.finfo(float).eps and b_norm <= np.finfo(float).eps:
        return cos, True 
    elif a_norm <= np.finfo(float).eps: 
        print_warn_log("All the data is Zero in npu dump data. Compare by relative error.")
        return get_max_rel_err(n_value, b_value)
    elif b_norm <= np.finfo(float).eps:
        print_warn_log("All the data is Zero in bench dump data. Compare by relative error.")
    else: 
        cos = num / (a_norm * b_norm)
        if np.isnan(cos):
            print_warn_log("Dump data has NaN when comparing with Cosine Similarity.")
        return cos, cos > 0.99


def compare_builtin_type(bench_out, npu_out):
    if bench_out != npu_out:
        return CompareConst.NAN, False 
    return 1.0, True 


def flatten_compare_result(result):
    flatten_result = [] 
    for result_i in result:
        if isinstance(result_i, list):
            flatten_result += flatten_compare_result(result_i)
        else:
            flatten_result.append(result_i)
    return flatten_result 


def compare_core(bench_out, npu_out, alg):
    if type(bench_out) != type(npu_out):
        raise ValueError("bench and npu output type is different")
    if isinstance(bench_out, (list, tuple)):
        compare_result, test_success = [], True
        if len(bench_out) != len(npu_out):
            raise ValueError("bench and npu output structure is different")
        for b_out_i, n_out_i in zip(bench_out, npu_out):
            compare_result_i, test_success_i = compare_core(b_out_i, n_out_i, alg)
            compare_result.append(compare_result_i)
            test_success = test_success and test_success_i
    elif isinstance(bench_out, dict):
        b_keys, n_keys = set(bench_out.keys()), set(npu_out.keys())
        if b_keys != n_keys:
            raise ValueError("bench and npu output dictionary keys are different")
        compare_result, test_success = compare_core(list(bench_out.values()), list(npu_out.values()))
    elif isinstance(bench_out, torch.Tensor):
        compare_result, test_success = compare_torch_tensor(bench_out, npu_out, alg)
    elif isinstance(bench_out, (bool, int, float, str)):
        compare_result, test_success = compare_builtin_type(bench_out, npu_out)
    elif bench_out is None:
        return 1.0, True
    else:
        raise NotImplementedError("Unexpected output type in compare_core: {}".format(type(bench_out)))
    if isinstance(compare_result, list):
        compare_result = flatten_compare_result(compare_result)
    return compare_result, test_success

    
