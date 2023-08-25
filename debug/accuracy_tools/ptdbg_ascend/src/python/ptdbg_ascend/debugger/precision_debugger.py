import os
from ..common.utils import Const, make_dump_path_if_not_exists, print_error_log, print_info_log
from ..dump.dump import DumpUtil, acc_cmp_dump, write_to_disk
from ..dump.utils import set_dump_path, set_dump_switch_print_info, generate_dump_path_str, \
        set_dump_switch_config, set_backward_input
from ..overflow_check.utils import OverFlowUtil
from ..overflow_check.overflow_check import overflow_check
from ..hook_module.register_hook import register_hook_core
from ..hook_module.hook_module import HOOKModule
from .debugger_config import DebuggerConfig


class PrecisionDebugger:
    first_start = True
    hook_func = None

    # 提供两种使用方式：逐个传参和构造config后传config，看哪种使用方式更受欢迎，之后只保留一种
    def __init__(self, dump_path=None, hook_name=None, rank=None, step=[0], config=None):
        if config is None:
            if dump_path is None or hook_name is None:
                err_msg = "You must provide dump_path and hook_name argument to PrecisionDebugger\
                                when config is not provided."
                raise Exception(err_msg)
            self.config = DebuggerConfig(dump_path, hook_name, rank, step)
        else:
            self.config = config
            print_info_log("Debugger gets config, it will override preceding arguments.")

        self.configure_hook = self.get_configure_hook(config.hook_name)
        self.configure_hook()
        DumpUtil.target_iter = config.step
        DumpUtil.target_rank = config.rank
        make_dump_path_if_not_exists(config.dump_path)
        set_dump_path(config.dump_path)
        if config.hook_name == "overflow_check":
            PrecisionDebugger.hook_func = overflow_check
        else:
            PrecisionDebugger.hook_func = acc_cmp_dump

    def get_configure_hook(self, hook_name):
        if hook_name == "dump":
            return self.configure_full_dump
        elif hook_name == "overflow_check":
            return self.configure_overflow_dump
        else:
            raise ValueError("hook name {} is not in ['dump', 'overflow_check']".format(hook_name))

    def configure_full_dump(self, mode='api_stack', scope=[], api_list=[], filter_switch=Const.ON,
            input_output_mode=[Const.ALL], acl_config=None, backward_input=[], summary_only=False):
        set_dump_switch_config(mode=mode, scope=scope, api_list=api_list,
                               filter_switch=filter_switch, dump_mode=input_output_mode, summary_only=summary_only)
        if mode == 'acl' and acl_config is None:
            raise ValueError("acl_config must be configured when mode is 'acl'")
        elif mode == 'acl' and acl_config is not None:
            DumpUtil.dump_config = acl_config
        if mode == 'acl' and 'backward' in scope and not backward_input:
            raise ValueError("backward_input must be configured when mode is 'acl' and scope contains 'backward'")
        elif mode == 'acl' and 'backward' in scope and backward_input:
            set_backward_input(backward_input)

    def configure_overflow_dump(self, mode="api", acl_config=None, overflow_nums=1):
        if mode == "acl":
            DumpUtil.dump_switch_mode = mode
            DumpUtil.dump_config = acl_config
            if acl_config is None:
                raise ValueError("acl_config must be configured when mode is 'acl'")
        if isinstance(overflow_nums, int):
            OverFlowUtil.overflow_nums = overflow_nums
        else:
            raise ValueError("overflow_nums must be int")

    @classmethod
    def start(cls):
        if cls.first_start:
            register_hook_core(cls.hook_func)
            cls.first_start = False
        DumpUtil.dump_switch = "ON"
        dump_path_str = generate_dump_path_str()
        set_dump_switch_print_info("ON", DumpUtil.dump_switch_mode, dump_path_str)

    @classmethod
    def stop(cls):
        DumpUtil.dump_switch = "OFF"
        dump_path_str = generate_dump_path_str()
        set_dump_switch_print_info("OFF", DumpUtil.dump_switch_mode, dump_path_str)
        write_to_disk()
