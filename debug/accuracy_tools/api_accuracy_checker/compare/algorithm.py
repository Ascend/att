# 定义比对算法及比对标准

import torch
import numpy as np
from api_accuracy_checker.compare.compare_utils import CompareConst, check_dtype_comparable
from api_accuracy_checker.common.utils import Const 


def compare_torch_tensor(cpu_output, npu_output, compare_alg):
    if not check_dtype_comparable(cpu_output, npu_output):
        return CompareConst.NAN, False, f"Bench out dtype is {cpu_output.dtype} but\
                 npu output dtype is {npu_output.dtype}, cannot compare."
    if cpu_output.dtype in [bool, np.uint8, np.int8, np.int16, np.uint16, np.uint32, np.int32, np.int64, np.uint64]:
        return compare_bool_tensor(cpu_output, npu_output)
    return compare_alg(cpu_output, npu_output)

def compare_bool_tensor(cpu_output, npu_output):
    cpu_shape = cpu_output.shape
    npu_shape = npu_output.shape
    if cpu_shape != npu_shape:
        return CompareConst.NAN, False, ""
    error_nums = (cpu_output != npu_output).sum()
    error_rate = float(error_nums / cpu_output.size)
    return error_rate, error_rate == 0, ""

def get_msg_and_handle_value(b_value, n_value):
    msg = ""
    if not isinstance(b_value, np.ndarray) or not isinstance(n_value, np.ndarray):
        msg = f"Max rel err only support numpy array! The actual type is {type(b_value)}, {type(n_value)}."
        return CompareConst.NAN, False, msg
    if b_value.shape != n_value.shape:
        msg = f"Shape of bench and npu outputs don't match. bench: {b_value.shape}, npu: {n_value.shape}."
        return CompareConst.NAN, False, msg

    if n_value.dtype in Const.FLOAT_TYPE:
        zero_mask = (n_value == 0)
        # 给0的地方加上eps防止除0
        n_value[zero_mask] += np.finfo(n_value.dtype).eps 
        # 根据n_value为0的位置给n_value也加上eps，否则两者都是0的情况下相对误差会是1
        b_value[zero_mask] += np.finfo(n_value.dtype).eps 
    else:
        # int type + float eps 会报错，所以这里要强转
        b_value, n_value = b_value.astype(float), n_value.astype(float)
        zero_mask = (n_value == 0)
        n_value[zero_mask] += np.finfo(float).eps 
        b_value[zero_mask] += np.finfo(float).eps 
    return b_value, n_value, msg

def get_max_rel_err(b_value, n_value):
    b_value, n_value, msg = get_msg_and_handle_value(b_value, n_value)
    rel_err = np.abs((n_value - b_value) / b_value).max()
    if n_value.dtype == np.float32:
        bool_result = rel_err < 0.0001
    else:
        bool_result = rel_err < 0.001
    return rel_err, bool_result, msg

def get_max_abs_err(b_value, n_value):
    b_value, n_value, msg = get_msg_and_handle_value(b_value, n_value)
    abs_err = np.abs(b_value - n_value).max()
    bool_result = abs_err < 0.001
    return abs_err, bool_result, msg

def get_rel_err_ratio_thousandth(b_value, n_value):
    return get_rel_err_ratio(b_value, n_value, 0.001)

def get_rel_err_ratio_ten_thousandth(b_value, n_value):
    ratio, bool_result, msg = get_rel_err_ratio(b_value, n_value, 0.0001)
    if n_value.dtype == np.float16:
        msg = f"This indicator is not used to evaluate {n_value.dtype} data"
        return ratio, True, msg
    return ratio, bool_result, msg

def get_rel_err_ratio(b_value, n_value, thresholding):
    b_value, n_value, msg = get_msg_and_handle_value(b_value, n_value)
    rel_errs = np.abs((n_value - b_value) / b_value)
    ratio = np.divide(np.sum(rel_errs < thresholding), np.size(rel_errs))
    bool_result = ratio > (1 - thresholding)
    return ratio, bool_result, msg

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
        return -1, False, msg
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
        n_value = n_value_max.astype(float) / n_value_max 
        b_value = b_value_max.astype(float) / b_value_max
        cos = np.dot(n_value, b_value) / (np.linalg.norm(n_value) * np.linalg.norm(b_value))
        if np.isnan(cos):
            msg = "Dump data has NaN when comparing with Cosine Similarity."
        return cos, cos > 0.99, msg

def compare_uint8_data(b_value, n_value):
    if (b_value == n_value).all():
        return 1, True
    else:
        return 0, False

def compare_builtin_type(bench_out, npu_out):
    if not isinstance(bench_out, (bool, int, float, str)):
        return CompareConst.NA, True, ""
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

# 本函数用alg比对bench_out 和npu_out，返回详细比对结果compare_result和标志比对是否通过的布尔变量test_success
def compare_core(bench_out, npu_out, alg):
    msg = ""
    if not isinstance(bench_out, type(npu_out)):
        return [(CompareConst.NAN, "bench and npu output type is different.")], False, CompareConst.NA, CompareConst.NA, CompareConst.NA
    if isinstance(bench_out, (list, tuple)):
        compare_result, test_success, bench_dtype, npu_dtype, shape = [], True, [], [], []
        if len(bench_out) != len(npu_out):
            return [(CompareConst.NAN, "bench and npu output structure is different")], False, CompareConst.NA, CompareConst.NA, CompareConst.NA
        for b_out_i, n_out_i in zip(bench_out, npu_out):
            compare_result_i, test_success_i, bench_dtype_i, npu_dtype_i, shape_i = compare_core(b_out_i, n_out_i, alg)
            compare_result.append(compare_result_i)
            test_success = test_success and test_success_i
            bench_dtype.append(bench_dtype_i)
            npu_dtype.append(npu_dtype_i)
            shape.append(shape_i)
    elif isinstance(bench_out, dict):
        b_keys, n_keys = set(bench_out.keys()), set(npu_out.keys())
        if b_keys != n_keys:
            compare_result, test_success, bench_dtype, npu_dtype, shape = [(CompareConst.NAN, "bench and npu output dict keys are different")], False, \
                CompareConst.NA, CompareConst.NA, CompareConst.NA
        compare_result, test_success, bench_dtype, npu_dtype, shape = compare_core(list(bench_out.values()), list(npu_out.values()), alg)
    elif isinstance(bench_out, torch.Tensor):
        copy_bench_out = bench_out.detach().clone()
        copy_npu_out = npu_out.detach().clone()
        bench_dtype = str(copy_bench_out.dtype)
        npu_dtype = str(copy_npu_out.dtype)
        shape = list(npu_out.shape)
        if copy_npu_out.dtype == torch.bfloat16:
            copy_bench_out = copy_bench_out.to(torch.float32)
            copy_npu_out = copy_npu_out.to(torch.float32)
        compare_result, test_success, msg = compare_torch_tensor(copy_bench_out.numpy(), copy_npu_out.cpu().numpy(), alg)
    elif isinstance(bench_out, (bool, int, float, str)):
        compare_result, test_success, msg = compare_builtin_type(bench_out, npu_out)
        bench_dtype = str(type(bench_out))
        npu_dtype = str(type(npu_out))
        shape = str(type(npu_out))
    elif bench_out is None:
        compare_result, test_success, msg = CompareConst.NA, True, "output is None"
        bench_dtype = CompareConst.NAN
        npu_dtype = CompareConst.NAN
        shape = CompareConst.NAN
    else:
        compare_result, test_success, msg = CompareConst.NA, True, "Unexpected output type \
                     in compare_core: {}".format(type(bench_out))
        bench_dtype = CompareConst.NAN
        npu_dtype = CompareConst.NAN
        shape = CompareConst.NAN
    if isinstance(compare_result, list):
        compare_result = flatten_compare_result(compare_result)
    else:
        compare_result = [(compare_result, msg)]
    if isinstance(bench_dtype, list):
        bench_dtype = flatten_compare_result(bench_dtype)
        npu_dtype = flatten_compare_result(npu_dtype)
        shape = flatten_compare_result(shape)
    else:
        bench_dtype = [bench_dtype]
        npu_dtype = [npu_dtype]
        shape = [shape]
    return compare_result, test_success, bench_dtype, npu_dtype, shape