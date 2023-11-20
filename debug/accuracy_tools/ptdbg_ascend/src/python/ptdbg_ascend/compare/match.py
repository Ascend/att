import os
import yaml
from ..common.file_check_util import FileOpen


class AtenIrMapping():
    def __init__(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.join(cur_path, "mapping.yaml")
        with FileOpen(yaml_path, 'r') as f:
            self.aten_mapping = yaml.safe_load(f)

    def match(self, op1, op2):
        if "Aten" in op1 and "Aten" not in op2:
            return self.match_op(op1, op2)
        else:
            return self.match_op(op2, op1)

    def match_op(self, aten_op, torch_op):
        aten_op_raw_name_overload = '_'.join(aten_op.split("_")[1:-3])
        aten_op_raw_name = aten_op_raw_name_overload.split('.')[0]
        torch_op_raw_name = '_'.join(torch_op.split("_")[1:-3]).lower()
        matching_op = self.aten_mapping.get(aten_op_raw_name)
        if matching_op is None:
            return False
        if matching_op.lower() == torch_op_raw_name:
            return True
        return False


graph_mapping = AtenIrMapping()









