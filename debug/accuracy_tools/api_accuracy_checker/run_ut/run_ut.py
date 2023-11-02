import argparse
import os
import copy
import sys
import time
import torch_npu
import yaml
import torch
from tqdm import tqdm
from api_accuracy_checker.run_ut.data_generate import gen_api_params, gen_args
from api_accuracy_checker.common.utils import print_info_log, print_warn_log, get_json_contents, api_info_preprocess, \
    print_error_log, check_file_or_directory_path, initialize_save_path, Const
from api_accuracy_checker.compare.compare import Comparator
from api_accuracy_checker.hook_module.wrap_tensor import TensorOPTemplate
from api_accuracy_checker.hook_module.wrap_functional import FunctionalOPTemplate
from api_accuracy_checker.hook_module.wrap_torch import TorchOPTemplate
from api_accuracy_checker.run_ut.ut_api_info import UtAPIInfo
from api_accuracy_checker.common.config import msCheckerConfig

from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileOpen, FileCheckConst, FileChecker, \
    change_mode, check_file_suffix



def init_environment():
    cur_path = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(cur_path, "../hook_module/support_wrap_ops.yaml")
    with FileOpen(yaml_path, 'r') as f:
        WrapFunctionalOps = yaml.safe_load(f).get('functional')
    for f in dir(torch.nn.functional):
        if f != "__name__":
            locals().update({f: getattr(torch.nn.functional, f)})


init_environment()


def exec_api(api_type, api_name, args, kwargs):
    if api_type == "Functional":
        functional_api = FunctionalOPTemplate(api_name, str, False)
        out = functional_api.forward(*args, **kwargs)
    if api_type == "Tensor":
        tensor_api = TensorOPTemplate(api_name, str, False)
        out = tensor_api.forward(*args, **kwargs)
    if api_type == "Torch":
        torch_api = TorchOPTemplate(api_name, str, False)
        out = torch_api.forward(*args, **kwargs)
    return out


def generate_npu_params(input_args, input_kwargs, need_backward):
    def recursive_arg_to_npu(arg_in):
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_npu(arg) for arg in arg_in)
        elif isinstance(arg_in, torch.Tensor):
            if need_backward and arg_in.requires_grad:
                arg_in = arg_in.clone().detach().to("npu").requires_grad_()
                temp_arg_in = arg_in * 1
                arg_in = temp_arg_in.type_as(arg_in)
                arg_in.retain_grad()
                return arg_in
            else:
                return arg_in.clone().detach().to("npu")
        else:
            return arg_in

    npu_args = recursive_arg_to_npu(input_args)
    npu_kwargs = {key: recursive_arg_to_npu(value) for key, value in input_kwargs.items()}
    return npu_args, npu_kwargs

def generate_cpu_params(input_args, input_kwargs, need_backward):
    first_dtype = None
    def recursive_arg_to_cpu(arg_in):
        nonlocal first_dtype
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_cpu(arg) for arg in arg_in)
        elif isinstance(arg_in, torch.Tensor):
            if need_backward and arg_in.requires_grad:
                if str(arg_in.dtype) in Const.RAISE_PRECISION.keys() and arg_in.dtype != first_dtype:
                    arg_in = arg_in.clone().type(eval(Const.RAISE_PRECISION[str(arg_in.dtype)])).detach().requires_grad_()
                    if first_dtype is None:
                        first_dtype = arg_in.dtype
                else:
                    arg_in = arg_in.clone().detach().requires_grad_()
                temp_arg_in = arg_in * 1
                arg_in = temp_arg_in.type_as(arg_in)
                arg_in.retain_grad()
                return arg_in
            else:
                if str(arg_in.dtype) in Const.RAISE_PRECISION.keys() and arg_in.dtype != first_dtype:
                    arg_in = arg_in.clone().type(eval(Const.RAISE_PRECISION[str(arg_in.dtype)])).detach()
                    if first_dtype is None:
                        first_dtype = arg_in.dtype
                    return arg_in
                return arg_in.clone().detach()
        else:
            return arg_in

    cpu_args = recursive_arg_to_cpu(input_args)
    cpu_kwargs = {key: recursive_arg_to_cpu(value) for key, value in input_kwargs.items()}
    return cpu_args, cpu_kwargs

def run_ut(forward_file, backward_file, out_path, save_error_data):
    print_info_log("start UT test")
    forward_content = get_json_contents(forward_file)
    backward_content = get_json_contents(backward_file)
    api_setting_dict = get_json_contents("torch_ut_setting.json")
    compare = Comparator(out_path)
    for api_full_name, api_info_dict in tqdm(forward_content.items()):
        try:
            data_info = run_torch_api(api_full_name, api_setting_dict, backward_content, api_info_dict)
            is_fwd_success, is_bwd_success = compare.compare_output(api_full_name,
                                                                    data_info.bench_out,
                                                                    data_info.npu_out,
                                                                    data_info.bench_grad_out,
                                                                    data_info.npu_grad_out)
            if save_error_data:
                do_save_error_data(api_full_name, data_info, is_fwd_success, is_bwd_success)
        except Exception as err:
            [_, api_name, _] = api_full_name.split("*")
            if "expected scalar type Long" in str(err):
                print_warn_log(f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                               f"'int32_to_int64' list in accuracy_tools/api_accuracy_check/common/utils.py file.")
            else:
                print_error_log(f"Run {api_full_name} UT Error: %s" % str(err))
            compare.write_summary_csv((api_full_name, "SKIP", "SKIP", str(err)))
    change_mode(compare.save_path, FileCheckConst.DATA_FILE_AUTHORITY)
    change_mode(compare.detail_save_path, FileCheckConst.DATA_FILE_AUTHORITY)
    compare.print_pretest_result()


def do_save_error_data(api_full_name, data_info, is_fwd_success, is_bwd_success):
    if not is_fwd_success or not is_bwd_success:
        api_full_name = api_full_name.replace("*", ".")
        for element in data_info.in_fwd_data_list:
            UtAPIInfo(api_full_name + '.forward.input', element)
        UtAPIInfo(api_full_name + '.forward.output.bench', data_info.bench_out)
        UtAPIInfo(api_full_name + '.forward.output.npu', data_info.npu_out)
        UtAPIInfo(api_full_name + '.backward.input', data_info.grad_in)
        UtAPIInfo(api_full_name + '.backward.output.bench', data_info.bench_grad_out)
        UtAPIInfo(api_full_name + '.backward.output.npu', data_info.npu_grad_out)



def run_torch_api(api_full_name, api_setting_dict, backward_content, api_info_dict):
    in_fwd_data_list = []
    [api_type, api_name, _] = api_full_name.split("*")
    args, kwargs, need_grad = get_api_info(api_info_dict, api_name)
    in_fwd_data_list.append(args)
    in_fwd_data_list.append(kwargs)
    need_backward = api_full_name in backward_content
    need_backward = need_backward and need_grad
    if not need_grad:
        print_warn_log("%s function with out=... arguments don't support automatic differentiation, skip backward." % api_full_name)
    cpu_args, cpu_kwargs = generate_cpu_params(args, kwargs, need_backward)
    npu_args, npu_kwargs = generate_npu_params(args, kwargs, need_backward)
    grad_out, npu_grad_out = None, None
    if kwargs.get("device"):
        del kwargs["device"]
    out = exec_api(api_type, api_name, cpu_args, cpu_kwargs)
    npu_out = exec_api(api_type, api_name, npu_args, npu_kwargs)
    grad_input_index = api_setting_dict.get(api_name)
    grad_index = None
    grad = None
    if grad_input_index is not None:
        grad_index = grad_input_index.get('grad_index')

    if need_backward:
        grad_out, npu_grad_out, grad, npu_grad = run_backward(api_full_name, cpu_args, backward_content, grad_index, npu_args,
                                                              npu_out, out)
    if grad_index is not None:
        return UtDataInfo(grad_out, npu_grad_out, npu_out[grad_index], out[grad_index], grad, in_fwd_data_list)
    return UtDataInfo(grad_out, npu_grad_out, npu_out, out, grad, in_fwd_data_list)


def get_api_info(api_info_dict, api_name):
    convert_type, api_info_dict = api_info_preprocess(api_name, api_info_dict)
    need_grad = True
    if api_info_dict.get("kwargs") and "out" in api_info_dict.get("kwargs"):
        need_grad = False
    args, kwargs = gen_api_params(api_info_dict, need_grad, convert_type)
    return args, kwargs, need_grad


def run_backward(api_full_name, args, backward_content, grad_index, npu_args, npu_out, out):
    backward_args = backward_content[api_full_name]
    grad = gen_args(backward_args)[0]
    cpu_grad, _ = generate_cpu_params(grad, {}, False)
    if grad_index is not None:
        out[grad_index].backward(cpu_grad)
    elif isinstance(out, (list, tuple)):
        raise NotImplementedError("Multiple backward is not supported.")
    else:
        out.backward(cpu_grad)
    args_grad = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            args_grad.append(arg.grad)
    grad_out = args_grad
    npu_grad = grad.clone().detach().npu()
    if grad_index is not None:
        npu_out[grad_index].backward(npu_grad)
    else:
        npu_out.backward(npu_grad)
    npu_args_grad = []
    for arg in npu_args:
        if isinstance(arg, torch.Tensor):
            npu_args_grad.append(arg.grad)
    npu_grad_out = npu_args_grad
    return grad_out, npu_grad_out, grad, npu_grad


def initialize_save_error_data():
    error_data_path_checker = FileChecker(msCheckerConfig.error_data_path, FileCheckConst.DIR,
                                          ability=FileCheckConst.WRITE_ABLE)
    error_data_path = error_data_path_checker.common_check()
    initialize_save_path(error_data_path, 'ut_error_data' + time.strftime("%Y%m%d%H%M%S"))


def _run_ut_parser(parser):
    parser.add_argument("-forward", "--forward_input_file", dest="forward_input_file", default="", type=str,
                        help="<Required> The api param tool forward result file: generate from api param tool, "
                             "a json file.",
                        required=True)
    parser.add_argument("-backward", "--backward_input_file", dest="backward_input_file", default="", type=str,
                        help="<Required> The api param tool backward result file: generate from api param tool, "
                             "a json file.",
                        required=True)
    parser.add_argument("-o", "--out_path", dest="out_path", default="", type=str,
                        help="<optional> The ut task result out path.",
                        required=False)
    parser.add_argument('-save_error_data', dest="save_error_data", action="store_true",
                        help="<optional> Save compare failed api output.", required=False)
    parser.add_argument("-j", "--jit_compile", dest="jit_compile", action="store_true",
                        help="<optional> whether to turn on jit compile", required=False)
    parser.add_argument("-d", "--device", dest="device_id", type=int, help="<optional> set NPU device id to run ut",
                        default=0, required=False)


def _run_ut():
    parser = argparse.ArgumentParser()
    _run_ut_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    torch.npu.set_compile_mode(jit_compile=args.jit_compile)
    npu_device = "npu:" + str(args.device_id)
    try:
        torch.npu.set_device(npu_device)
    except Exception:
        print_error_log(f"Set NPU device id failed. device id is: {args.device_id}")
        raise NotImplementedError
    forward_file = os.path.realpath(args.forward_input_file)
    backward_file = os.path.realpath(args.backward_input_file)
    check_file_suffix(forward_file, FileCheckConst.JSON_SUFFIX)
    check_file_suffix(backward_file, FileCheckConst.JSON_SUFFIX)
    out_path = os.path.realpath(args.out_path) if args.out_path else "./"
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    save_error_data = args.save_error_data
    if save_error_data:
        initialize_save_error_data()
    run_ut(forward_file, backward_file, out_path, save_error_data)


class UtDataInfo:
    def __init__(self, bench_grad_out, npu_grad_out, npu_out, bench_out, grad_in, in_fwd_data_list):
        self.bench_grad_out = bench_grad_out
        self.npu_grad_out = npu_grad_out
        self.npu_out = npu_out
        self.bench_out = bench_out
        self.grad_in = grad_in
        self.in_fwd_data_list = in_fwd_data_list


if __name__ == '__main__':
    _run_ut()
    print_info_log("UT task completed.")
