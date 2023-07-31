# 定义API INFO，保存基本信息，用于后续结构体的落盘，注意考虑random场景及真实数据场景


class APIInfo:
    def __init__(self, api_name):
        self.api_name = api_name

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
            stack_line = " ".join([
                "File", ", ".join([path, " ".join(["line", str(line)]), " ".join(["in", func]),
                                " ".join(["\n", code[0].strip() if code else code])])])
            stack_str.append(stack_line)
        self.stack_info_struct = {self.api_name: stack_str}
    

class BackwardAPIInfo(APIInfo):
    def __init__(self, name, grads):
        super().__init__(name)
        self.analyze_api_input(grads)
    
    def analyze_api_input(self, grads):
        grads_info_list = self.analyze_element(grads)
        self.grad_info_struct = {self.api_name:grads_info_list}

