import argparse
import os
import sys
sys.path.append("..")
import yaml
import torch
from data_generate import gen_api_params, gen_args
from common.utils import print_info_log, get_json_contents
from compare.compare import Comparator

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "../hook_module/support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapFunctionalOps = yaml.safe_load(f).get('functional')

for f in dir(torch.nn.functional):
    if f != "__name__":
        locals().update({f: getattr(torch.nn.functional, f)})


def exec_api(api_type, api_name, args, kwargs):
    if api_type == "Functional":
        out = eval(api_name)(*args, **kwargs)
    if api_type == "Tensor":
        out = getattr(torch._C._TensorBase, str(api_name))(*args, **kwargs)
    if api_type == "Torch":
        out = getattr(torch._C._VariableFunctionsClass, str(api_name))(*args, **kwargs)
    return out


def generate_npu_params(cpu_args, cpu_kwargs, need_backward):
    npu_args = []
    npu_kwargs = {}
    if need_backward:
        for arg_in in cpu_args:
            arg_in = arg_to_npu(arg_in)
            npu_args.append(arg_in)
        for key, value in cpu_kwargs.items():
            value = arg_to_npu(value)
            npu_kwargs[key] = value
    else:
        for arg_in in cpu_args:
            if isinstance(arg_in, torch.Tensor):
                arg_in = arg_in.clone().detach().to("npu")
            npu_args.append(arg_in)
        for key, value in cpu_kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.clone().detach().to("npu")
            npu_kwargs[key] = value
    return npu_args, npu_kwargs


def arg_to_npu(arg_in):
    if isinstance(arg_in, torch.Tensor) and arg_in.dtype in [torch.float, torch.float16,
                                                             torch.float64] and arg_in.requires_grad:
        arg_in = arg_in.clone().detach().to("npu").requires_grad_()
    elif isinstance(arg_in, torch.Tensor):
        arg_in = arg_in.clone().detach().to("npu")
    return arg_in


def run_ut(forward_file, backward_file, out_path, save_error_data):
    print_info_log("start UT test")
    forward_content = get_json_contents(forward_file)
    backward_content = get_json_contents(backward_file)
    api_setting_dict = get_json_contents("torch_ut_setting.json")
    compare = Comparator(out_path)
    for api_full_name, api_info_dict in forward_content.items():
        grad_out, npu_grad_out, npu_out, out = run_torch_api(api_full_name, api_setting_dict, backward_content,
                                                             api_info_dict)
        compare.compare_output(api_full_name, out, npu_out, grad_out, npu_grad_out)

    compare.print_pretest_result()
    compare.write_compare_csv()


def run_torch_api(api_full_name, api_setting_dict, backward_content, value):
    [api_type, api_name, _] = api_full_name.split("*")
    args, kwargs = gen_api_params(value, api_name[-1] != "_")
    inplace = kwargs.get("inplace") if kwargs.get("inplace") else None
    need_backward = api_full_name in backward_content and api_name[-1] != "_" and inplace is not True
    npu_args, npu_kwargs = generate_npu_params(args, kwargs, need_backward)
    grad_out, npu_grad_out = None, None
    out = exec_api(api_type, api_name, args, kwargs)
    npu_out = exec_api(api_type, api_name, npu_args, npu_kwargs)
    grad_input_index = api_setting_dict.get(api_name)
    grad_index = None
    if grad_input_index is not None:
        grad_index = grad_input_index.get('grad_index')

    if need_backward:
        backward_args = backward_content[api_full_name]
        grad = gen_args(backward_args)[0]
        if grad_index is not None:
            out[grad_index].backward(grad)
        else:
            out.backward(grad)
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
    return grad_out, npu_grad_out, npu_out, out


def _run_ut_parser(parser):
    parser.add_argument("-forward", "--forward_input_file", dest="forward_input_file", default="",
                        help="<Required> The api param tool forward result file: generate from api param tool, "
                             "a json file.",
                        required=True)
    parser.add_argument("-backward", "--backward_input_file", dest="backward_input_file", default="",
                        help="<Required> The api param tool backward result file: generate from api param tool, "
                             "a json file.",
                        required=True)
    parser.add_argument("-o", "--out_path", dest="out_path", default="",
                        help="<optional> The ut task result out path.",
                        required=False)
    parser.add_argument('-save_error_data', dest="save_error_data", action="store_true",
                        help="<optional> Save compare failed api output.", required=False)
    parser.add_argument("-c", "--jit_compile", dest="jit_compile", help="<optional> whether to turn on jit compile",
                        default=True, required=False)


def _run_ut():
    parser = argparse.ArgumentParser()
    _run_ut_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    if not args.jit_compile:
        torch.npu.set_compile_mode(jit_compile=False)
    forward_file = os.path.realpath(args.forward_input_file)
    backward_file = os.path.realpath(args.backward_input_file)
    if not forward_file.endswith(".json") or not backward_file.endswith(".json"):
        raise ValueError("The forward_input_file and backward_input_file should be a json file!")
    out_path = os.path.realpath(args.out_path) if args.out_path else "./"
    save_error_data = args.save_error_data
    run_ut(forward_file, backward_file, out_path, save_error_data)


if __name__ == '__main__':
    _run_ut()
    print_info_log("UT task completed.")