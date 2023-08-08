import argparse
import os
import sys
import torch_npu
import yaml
import torch
from api_accuracy_checker.run_ut.data_generate import gen_api_params, gen_args
from api_accuracy_checker.common.utils import print_info_log, print_warn_log, get_json_contents, check_need_convert, \
    print_error_log
from api_accuracy_checker.compare.compare import Comparator
from api_accuracy_checker.hook_module.wrap_tensor import TensorOPTemplate
from api_accuracy_checker.hook_module.wrap_functional import FunctionalOPTemplate
from api_accuracy_checker.hook_module.wrap_torch import TorchOPTemplate

NO_GRAD_APIS = ["hardtanh"]

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "../hook_module/support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapFunctionalOps = yaml.safe_load(f).get('functional')

for f in dir(torch.nn.functional):
    if f != "__name__":
        locals().update({f: getattr(torch.nn.functional, f)})


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


def generate_npu_params(cpu_args, cpu_kwargs, need_backward):
    def recursive_arg_to_npu(arg_in):
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_npu(arg) for arg in arg_in)
        elif isinstance(arg_in, torch.Tensor):
            if need_backward and arg_in.requires_grad:
                return arg_in.clone().detach().to("npu").requires_grad_()
            else:
                return arg_in.clone().detach().to("npu")
        else:
            return arg_in

    npu_args = recursive_arg_to_npu(cpu_args)
    npu_kwargs = {key: recursive_arg_to_npu(value) for key, value in cpu_kwargs.items()}
    return npu_args, npu_kwargs


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
    convert_type = check_need_convert(api_name)
    need_grad = True
    if value.get("kwargs") and "out" in value.get("kwargs"):
        need_grad = False
    if api_name[-1] == "_" or api_name in NO_GRAD_APIS:
        need_grad = False
    args, kwargs = gen_api_params(value, need_grad, convert_type)
    inplace = kwargs.get("inplace") if kwargs.get("inplace") else None
    need_backward = api_full_name in backward_content and api_name[-1] != "_" and inplace is not True
    need_backward = need_backward and need_grad
    if inplace or not need_grad:
        print_warn_log("%s involves in-place operations, skip backward" % api_full_name)
    npu_args, npu_kwargs = generate_npu_params(args, kwargs, need_backward)
    grad_out, npu_grad_out = None, None
    if kwargs.get("device"):
        del kwargs["device"]
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
        elif isinstance(out, (list, tuple)):
            raise NotImplementedError("Multiple backward is not supported.")
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
    if grad_index is not None:
        return grad_out, npu_grad_out, npu_out[grad_index], out[grad_index]
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
                        default=False, required=False)
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
    if not forward_file.endswith(".json") or not backward_file.endswith(".json"):
        raise ValueError("The forward_input_file and backward_input_file should be a json file!")
    out_path = os.path.realpath(args.out_path) if args.out_path else "./"
    save_error_data = args.save_error_data
    run_ut(forward_file, backward_file, out_path, save_error_data)


if __name__ == '__main__':
    _run_ut()
    print_info_log("UT task completed.")
