import os
from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.common.base_api import BaseAPIInfo
from api_accuracy_checker.common.utils import write_pt, create_directory
from ptdbg_ascend.src.python.ptdbg_ascend.common.utils import check_path_before_create


class UtAPIInfo(BaseAPIInfo):
    def __init__(self, api_name, element, is_forward=True, is_save_data=True, save_path=msCheckerConfig.error_data_path, 
                 forward_path='ut_error_data', backward_path='ut_error_data'):
        super().__init__(api_name, is_forward, is_save_data, save_path, forward_path, backward_path)
        self.analyze_element(element)

    def analyze_tensor(self, arg):
        single_arg = {}

        api_args = self.api_name + '.' + str(self.args_num)
        if self.is_forward:
            forward_real_data_path = os.path.join(self.save_path, self.forward_path)
            check_path_before_create(forward_real_data_path)
            create_directory(forward_real_data_path)
            file_path = os.path.join(forward_real_data_path, f'{api_args}.pt')
        else:
            backward_real_data_path = os.path.join(self.save_path, self.backward_path)
            check_path_before_create(backward_real_data_path)
            create_directory(backward_real_data_path)
            file_path = os.path.join(backward_real_data_path, f'{api_args}.pt')
        self.args_num += 1
        pt_path = write_pt(file_path, arg.contiguous().cpu().detach())
        single_arg.update({'type': 'torch.Tensor'})
        single_arg.update({'datapath': pt_path})
        single_arg.update({'requires_grad': arg.requires_grad})

        return single_arg
