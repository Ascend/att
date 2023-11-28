import os
from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.common.base_api import BaseAPIInfo
from api_accuracy_checker.common.utils import write_pt, create_directory
from ptdbg_ascend.src.python.ptdbg_ascend.common.utils import check_path_before_create


class UtAPIInfo(BaseAPIInfo):
    def __init__(self, api_name, element, ut_error_data_dir):
        super().__init__(api_name, True, True, msCheckerConfig.error_data_path, '', '')
        self.ut_error_data_dir = ut_error_data_dir
        self.analyze_element(element)

    def analyze_tensor(self, arg):
        single_arg = {}
        api_args = self.api_name + '.' + str(self.args_num)
        ut_error_data_path = os.path.join(self.save_path, self.ut_error_data_dir)
        check_path_before_create(ut_error_data_path)
        create_directory(ut_error_data_path)
        file_path = os.path.join(ut_error_data_path, f'{api_args}.pt')
        self.args_num += 1
        pt_path = write_pt(file_path, arg.contiguous().cpu().detach())
        single_arg.update({'type': 'torch.Tensor'})
        single_arg.update({'datapath': pt_path})
        single_arg.update({'requires_grad': arg.requires_grad})
        return single_arg
