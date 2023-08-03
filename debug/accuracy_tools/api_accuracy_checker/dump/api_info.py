# 定义API INFO，保存基本信息，用于后续结构体的落盘，注意考虑random场景及真实数据场景
import inspect
import torch
import torch_npu
from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.common.utils import print_error_log
from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.dump.utils import write_npy

class APIInfo:
    def __init__(self, api_name):
        self.rank = torch_npu.npu.current_device()
        self.api_name = api_name
        self.save_real_data = msCheckerConfig.real_data

    def analyze_element(self, element):
        if isinstance(element, (list, tuple)):
            out = []
            for item in element:
                out.append(self.analyze_element(item))
        elif isinstance(element, dict):
            out = {}
            for key, value in element.items():
                out[key] = self.analyze_element(value)

        elif isinstance(element, torch.Tensor):
            out = self.analyze_tensor(element, self.save_real_data)

        elif self.is_builtin_class(element):
            out = self.analyze_builtin(element)
        else:
            msg = f"Type {type(element)} is unsupported at analyze_element"
            print_error_log(msg)

            raise NotImplementedError(msg)
        return out


    def analyze_tensor(self, arg, save_real_data):
        single_arg = {}
        if not save_real_data:
            
            single_arg.update({'type' : 'torch.Tensor'})
            single_arg.update({'dtype' : str(arg.dtype)})
            single_arg.update({'shape' : arg.shape})
            single_arg.update({'Max' : self.transfer_types(self.get_tensor_extremum(arg,'max'), str(arg.dtype))})
            single_arg.update({'Min' : self.transfer_types(self.get_tensor_extremum(arg,'min'), str(arg.dtype))})
            single_arg.update({'requires_grad': arg.requires_grad})
            
        else:
            dump_path = msCheckerConfig.dump_path
            real_data_path = os.path.join(dump_path, 'real_data')
            file_path = os.path.join(real_data_path, self.api_name)
            npy_path = write_npy(file_path, arg.contiguous().cpu().detach().numpy())
            single_arg.update({'type' : 'torch.Tensor'})
            single_arg.update({'datapath' : npy_path})
            single_arg.update({'requires_grad': arg.requires_grad})
        return single_arg

    def analyze_builtin(self, arg):
        single_arg = {}
        if isinstance(arg, slice):
            single_arg.update({'type' : "slice"})
            single_arg.update({'value' : [arg.start, arg.stop, arg.step]})
        else:
            single_arg.update({'type' : self.get_type_name(str(type(arg)))})
            single_arg.update({'value' : arg})
        return single_arg

    def transfer_types(self, data, dtype):
        if 'int' in dtype or 'bool' in dtype:
            return int(data)
        else:
            return float(data)

    def is_builtin_class(self, element):
        if element is None or isinstance(element, (bool,int,float,str,slice)):
            return True
        return False

    
    def get_tensor_extremum(self, data, operator):
        if data.dtype is torch.bool:
            if operator == 'max':
                return True in data
            elif operator == 'min':
                return False not in data
        if operator == 'max':
            return torch._C._VariableFunctionsClass.max(data).item()
        else:
            return torch._C._VariableFunctionsClass.min(data).item()
    
    def get_type_name(self, name):

        left = name.index("'")
        right = name.rindex("'")
        return name[left + 1 : right]



class ForwardAPIInfo(APIInfo):
    def __init__(self, name, args, kwargs):
        super().__init__(name)
        self.analyze_api_input(args, kwargs)
        self.analyze_api_call_stack() 
    
    def analyze_api_input(self, args, kwargs):
        args_info_list = self.analyze_element(args)
        kwargs_info_dict = self.analyze_element(kwargs)
        self.api_info_struct = {self.api_name: {"args":args_info_list, "kwargs":kwargs_info_dict}}

    def analyze_api_call_stack(self):
        stack_str = []
        for (_, path, line, func, code, _) in inspect.stack()[3:]:
            if not code: continue
            stack_line = " ".join([
                "File", ", ".join([path, " ".join(["line", str(line)]), " ".join(["in", func]),
                                " ".join(["\n", code[0].strip()])])])
            stack_str.append(stack_line)
        self.stack_info_struct = {self.api_name: stack_str}
    

class BackwardAPIInfo(APIInfo):
    def __init__(self, name, grads):
        super().__init__(name)
        self.analyze_api_input(grads)
    
    def analyze_api_input(self, grads):
        grads_info_list = self.analyze_element(grads)
        self.grad_info_struct = {self.api_name:grads_info_list}
