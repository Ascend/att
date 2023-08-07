# 定义比对算法及比对标准

import torch
import numpy as np
from api_accuracy_checker.compare.compare_utils import CompareConst

def compare_torch_tensor(cpu_output, npu_output, compare_alg):
    if cpu_output.dtype == torch.bool or cpu_output.dtype == torch.uint8:
        if npu_output.dtype != cpu_output.dtype:
            return CompareConst.NAN, False, f"Bench out dtype is {cpu_output.dtype} but npu output dtype is {npu_output.dtype}, cannot compare."
        return compare_bool_tensor(cpu_output, npu_output)
    return compare_alg(cpu_output.detach().numpy(), npu_output.detach().numpy())


def compare_bool_tensor(cpu_output, npu_output):
    error_rate = CompareConst.NAN
    cpu_shape = cpu_output.shape
    npu_shape = npu_output.shape
    if cpu_shape != npu_shape:
        return error_rate, False, ""
    npu_data = npu_output.cpu().detach().numpy()
    bench_data = cpu_output.detach().numpy()
    data_size = bench_data.size
    error_nums = (bench_data != npu_data).sum()
    error_rate = float(error_nums / data_size)
    return error_rate, error_rate < 0.001, ""


def get_max_rel_err(n_value, b_value):
    msg = ""
    if not isinstance(n_value, np.ndarray) or not isinstance(b_value, np.ndarray):
        msg = f"Max rel err only support numpy array! The actual type is {type(n_value)}, {type(b_value)}."
        return CompareConst.NAN, False, msg
    if n_value.shape != b_value.shape:
        msg = f"Shape of npu and bench outputs don't match. NPU: {n_value.shape}, bench: {b_value.shape}."
        return CompareConst.NAN, False, msg
    if n_value.dtype != b_value.dtype:
        msg = f"Dtype of npu and bench outputs don't match. NPU: {n_value.dtype}, bench: {b_value.dtype}."

    rel_err = np.abs((n_value - b_value) / (b_value + np.finfo(b_value.dtype).eps)).max()
    bool_result = rel_err < 0.001
    
    return rel_err, bool_result, msg


def max_rel_err_standard(max_rel_errs):
    bool_result = np.array(max_rel_errs) < 0.001 
    return np.all(bool_result), bool_result


def cosine_standard(compare_result):
    bool_result = np.array(compare_result) > 0.99
    return np.all(bool_result), bool_result


def cosine_sim(cpu_output, npu_output):
    msg = ""
    n_value = npu_output.reshape(-1)
    b_value = cpu_output.reshape(-1)
    cos = CompareConst.NA
    np.seterr(divide="ignore", invalid="ignore")
    if n_value.shape != b_value.shape:
        msg = f"Shape of npu and bench outputs don't match. NPU: {n_value.shape}, bench: {b_value.shape}."
        return  -1, False, msg
    if len(n_value) == 1:
        msg = "All the data in npu dump data is scalar. Please refer to other compare algorithms."
        return cos, True, msg 
    n_value_max = np.max(np.abs(n_value))
    b_value_max = np.max(np.abs(b_value))
    if n_value_max <= np.finfo(float).eps and b_value_max <= np.finfo(float).eps:
        return cos, True, msg 
    elif n_value_max <= np.finfo(float).eps:
        msg = "All the data is zero in npu dump data."
        return CompareConst.NAN, False, msg 
    elif b_value_max <= np.finfo(float).eps:
        msg = "All the data is zero in bench dump data."
        return CompareConst.NAN, False, msg 
    else:
        n_value /= n_value_max 
        b_value /= b_value_max 
        cos = np.dot(n_value, b_value) / (np.linalg.norm(n_value) * np.linalg.norm(b_value))
        if np.isnan(cos):
            msg = "Dump data has NaN when comparing with Cosine Similarity."
        return cos, cos > 0.99, msg


def compare_uint8_data(n_value, b_value):
    if (n_value == b_value).all():
        return 1, True
    else:
        return 0, False


def compare_builtin_type(bench_out, npu_out):
    if not isinstance(bench_out, (bool, int, float, str)):
        return CompareConst.NA, True, f"The data is not builtin type: {type(bench_out)}"
    if bench_out != npu_out:
        return CompareConst.NAN, False, ""
    return True, True, ""


def flatten_compare_result(result):
    flatten_result = []
    for result_i in result:
        if isinstance(result_i, list):
            flatten_result += flatten_compare_result(result_i)
        else:
            flatten_result.append(result_i)
    return flatten_result


def compare_core(bench_out, npu_out, alg):
    msg = ""
    if type(bench_out) != type(npu_out):
        compare_result, test_success, msg = CompareConst.NAN, False, "bench and npu output type is different."
    if isinstance(bench_out, (list, tuple)):
        compare_result, test_success = [], True
        if len(bench_out) != len(npu_out):
            compare_result, test_success, msg = CompareConst.NAN, False, "bench and npu output structure is different"
        for b_out_i, n_out_i in zip(bench_out, npu_out):
            compare_result_i, test_success_i = compare_core(b_out_i, n_out_i, alg)
            compare_result.append(compare_result_i)
            test_success = test_success and test_success_i
    elif isinstance(bench_out, dict):
        b_keys, n_keys = set(bench_out.keys()), set(npu_out.keys())
        if b_keys != n_keys:
            compare_result, test_success, msg = CompareConst.NAN, False, "bench and npu output dict keys are different"
        compare_result, test_success = compare_core(list(bench_out.values()), list(npu_out.values()))
    elif isinstance(bench_out, torch.Tensor):
        compare_result, test_success, msg = compare_torch_tensor(bench_out, npu_out, alg)
    elif isinstance(bench_out, (bool, int, float, str)):
        compare_result, test_success, msg = compare_builtin_type(bench_out, npu_out)
    elif bench_out is None:
        compare_result, test_success, msg = CompareConst.NA, True, "output is None"
    else:
        compare_result, test_success, msg = CompareConst.NA, True, "Unexpected output type \
                     in compare_core: {}".format(type(bench_out))
    if isinstance(compare_result, list):
        compare_result = flatten_compare_result(compare_result)
    else:
        compare_result = [(compare_result, test_success, msg)]
    return compare_result, test_success


