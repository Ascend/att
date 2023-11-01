import os
import torch.nn as nn
from ..common.utils import print_error_log, DumpException
from .dump import acc_cmp_dump
from ..hook_module.api_registry import api_register

module_count = {}


def module_dump(module, dump_name):
    if not isinstance(module, nn.Module):
        print_error_log("The parameter:module in module_dump is not a Module subclass.")
        raise DumpException(DumpException.INVALID_PARAM_ERROR)
    if not isinstance(dump_name, str):
        print_error_log("The parameter:dump_name in module_dump is not a str type.")
        raise DumpException(DumpException.INVALID_PARAM_ERROR)
    pid = os.getpid()
    api_register.api_originality()
    if dump_name not in module_count:
        module_count[dump_name] = 1
        dump_name += '_0_'
    else:
        module_count[dump_name] += 1
        dump_name = dump_name + '_' + str(module_count[dump_name] - 1)
    module.register_forward_hook(acc_cmp_dump(dump_name + "forward", pid=pid))
    module.register_backward_hook(acc_cmp_dump(dump_name + "backward", pid=pid))


def module_dump_end():
    api_register.api_modularity()
