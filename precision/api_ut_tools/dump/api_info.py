# 定义API INFO，保存基本信息，用于后续结构体的落盘，注意考虑random场景及真实数据场景


class APIInfo:
    def __init__(self, api_name):
        self.api_name = api_name
