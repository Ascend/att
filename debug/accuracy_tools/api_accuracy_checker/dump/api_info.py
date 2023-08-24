# 定义API INFO，保存基本信息，用于后续结构体的落盘，注意考虑random场景及真实数据场景
import inspect
from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.common.config import BaseAPIInfo


class APIInfo(BaseAPIInfo):
    def __init__(self, api_name, is_forward, is_save_data=msCheckerConfig.real_data, 
                 save_path=msCheckerConfig.dump_path, forward_path='forward_real_data', 
                 backward_path='backward_real_data'):
        super().__init__(api_name, is_forward, is_save_data, save_path, forward_path, backward_path)


class ForwardAPIInfo(APIInfo):
    def __init__(self, name, args, kwargs):
        super().__init__(name, is_forward=True)
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
        super().__init__(name, is_forward=False)
        self.analyze_api_input(grads)
    
    def analyze_api_input(self, grads):
        grads_info_list = self.analyze_element(grads)
        self.grad_info_struct = {self.api_name:grads_info_list}
