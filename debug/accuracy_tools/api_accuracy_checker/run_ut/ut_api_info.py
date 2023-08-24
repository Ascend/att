from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.common.config import BaseAPIInfo


class UtAPIInfo(BaseAPIInfo):
    def __init__(self, api_name, element, is_forward=True, is_save_data=True, save_path=msCheckerConfig.error_data_path, 
                 forward_path='ut_error_data', backward_path='ut_error_data'):
        super().__init__(api_name, is_forward, is_save_data, save_path, forward_path, backward_path)
        self.analyze_element(element)
