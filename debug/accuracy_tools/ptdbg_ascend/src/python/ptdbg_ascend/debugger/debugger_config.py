import os
from ..common.utils import print_warn_log


class DebuggerConfig:
    def __init__(self, dump_path, hook_name, rank=None, step=[0]):
        self.dump_path = dump_path
        self.hook_name = hook_name
        self.rank = rank
        self.step = step
        if self.step:
            self.step.sort()
        self.check()

    def check(self):
        dump_root = os.path.split(self.dump_path)[0]
        if not os.path.exists(dump_root):
            raise ValueError("dump path {} does not exist".format(dump_root))
        if self.hook_name not in ["dump", "overflow_check"]:
            raise ValueError("hook_name should be in ['dump', 'overflow_check']".format(self.hook_name))
        if self.rank is not None and not isinstance(self.rank, int):
            raise ValueError("rank {} should be int".format(self.rank))
        elif isinstance(self.rank, int):
            print_warn_log(f"Rank argument is provided. Only rank {self.rank} data will be dumpped.")
        if not isinstance(self.step, list):
            raise ValueError("step {} should be list".format(self.step))
        if len(self.step) == 0:
            raise ValueError("step {} should not be empty".format(self.step))
        for s in self.step:
            if not isinstance(s, int):
                raise ValueError("step element {} should be int".format(s))
        return True

