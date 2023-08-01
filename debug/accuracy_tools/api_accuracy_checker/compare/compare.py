# 进行比对及结果展示
import os 
from prettytable import Prettytable 
from .algorithm import compare_core, cosine_sim, cosine_standard 
from ..common.utils import get_json_contents, print_error_log, print_info_log, write_csv
from .compare_utils import CompareConst 

class Comparator:
    TEST_FILE_NAME = "pretest_result.csv"
    # consts for result csv 
    COLUMN_API_NAME = "API name"
    COLUMN_FORWARD_SUCCESS = "Forward Test Success"
    COLUMN_BACKWARD_SUCCESS = "Backward Test Success"
    COLUMN_STACK_INFO = "Traceback callstack info"

    def __init__(self, result_save_path, stack_info_json_path=None):
        self.save_path = os.path.join(result_save_path, self.TEST_FILE_NAME)
        if stack_info_json_path:
            self.stack_info = get_json_contents(stack_info_json_path)
        else:
            self.stack_info = None 
        self.compare_alg = {}
        self.compare_alg_names = [] 
        self.register_compare_algorithm("Cosine Similarity", cosine_sim, cosine_standard) 
        self.test_results = []
        self.test_result_cnt = {"forward_fail_num":0, "backward_fail_num":0, "forward_and_backward_fail_num":0, "success_num":0}

    def print_pretest_result(self):
        res_dict = {
            "forward_not_pass": self.test_result_cnt['forward_fail_num'],
            "backward_not_pass": self.test_result_cnt['backward_fail_num'],
            "forward_and_backward_not_pass": self.test_result_cnt['forward_and_backward_fail_num'],
            "pass": self.test_result_cnt['success_num']
        }    
        tb = Prettytable()
        tb.add_column("Category", list(res_dict.keys()))
        tb.add_column("statistics",list(res_dict.values()))
        info_tb = str(tb)
        print_info_log(info_tb)

    def write_compare_csv(self):
        self.write_summary_csv() 

    def write_summary_csv(self):
        test_rows = [[self.COLUMN_API_NAME, self.COLUMN_FORWARD_SUCCESS, self.COLUMN_BACKWARD_SUCCESS]]
        if self.stack_info:
            test_rows[0].append(self.COLUMN_STACK_INFO)
        for result in self.test_results:
            name = result[0] 
            df_row = list(result[:3])
            if self.stack_info:
                stack_info = "\n".join(self.stack_info[name])
                df_row.append(stack_info)
            test_rows.append(df_row)
        write_csv(test_rows, self.save_path)

    def record_results(self, *args):
        self.test_results.append(args)

    def register_compare_algorithm(self, name, compare_func, standard):
        self.compare_alg.update({name: (compare_func, standard)})
        self.compare_alg_names.append(name)

    def compare_output(self, api_name, bench_out, npu_out, bench_grad=None, npu_grad=None):
        is_fwd_success, fwd_compare_alg_results = self._compare_core_wrapper(bench_out, npu_out)
        if bench_grad and npu_grad:
            is_bwd_success, bwd_compare_alg_results = self._compare_core_wrapper(bench_grad, npu_grad)
        else:
            is_bwd_success, bwd_compare_alg_results = CompareConst.NA, None 
        self.record_results(api_name, is_fwd_success, is_bwd_success, fwd_compare_alg_results, bwd_compare_alg_results)
        if is_fwd_success and is_bwd_success:
            self.test_result_cnt['success_num'] += 1
        elif not is_fwd_success and not is_bwd_success:
            self.test_result_cnt['forward_and_backward_fail_num'] += 1
        elif not is_fwd_success:
            self.test_result_cnt['forward_fail_num'] += 1
        else:
            self.test_result_cnt['backward_fail_num'] += 1

    def _compare_core_wrapper(self, bench_out, npu_out):
        name = self.compare_alg_names[0]
        detailed_result, test_success = compare_core(bench_out, npu_out, self.compare_lag[name][0])
        return test_success, detailed_result 