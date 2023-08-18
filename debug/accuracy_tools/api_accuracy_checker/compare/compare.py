# 进行比对及结果展示
import os
from prettytable import PrettyTable
from api_accuracy_checker.compare.algorithm import compare_core, cosine_sim, cosine_standard, get_max_rel_err, \
    compare_builtin_type, get_rel_err_ratio_thousandth, get_rel_err_ratio_ten_thousandth
from api_accuracy_checker.common.utils import get_json_contents, print_error_log, print_info_log, write_csv
from api_accuracy_checker.compare.compare_utils import CompareConst 


class Comparator:
    TEST_FILE_NAME = "pretest_result.csv"
    DETAIL_TEST_FILE_NAME = "pretest_details.csv"
    # consts for result csv 
    COLUMN_API_NAME = "API name"
    COLUMN_FORWARD_SUCCESS = "Forward Test Success"
    COLUMN_BACKWARD_SUCCESS = "Backward Test Success"
    COLUMN_STACK_INFO = "Traceback callstack info"

    def __init__(self, result_save_path, stack_info_json_path=None):
        self.save_path = os.path.join(result_save_path, self.TEST_FILE_NAME)
        self.detail_save_path = os.path.join(result_save_path, self.DETAIL_TEST_FILE_NAME)
        if stack_info_json_path:
            self.stack_info = get_json_contents(stack_info_json_path)
        else:
            self.stack_info = None
        self.compare_alg = {}
        self.register_compare_algorithm("Cosine Similarity", cosine_sim, cosine_standard)
        self.register_compare_algorithm("Max Relative Error", get_max_rel_err, None)
        self.register_compare_algorithm("Thousandth Relative Error Ratio", get_rel_err_ratio_thousandth, None)
        self.register_compare_algorithm("Ten Thousandth Relative Error Ratio", get_rel_err_ratio_ten_thousandth, None)
        self.register_compare_algorithm("Default: isEqual", compare_builtin_type, None)
        self.test_results = []
        self.test_result_cnt = {
            "forward_fail_num": 0, "backward_fail_num": 0, "forward_and_backward_fail_num": 0, "success_num": 0
        }
        self.result_save_path = result_save_path

    def print_pretest_result(self):
        res_dict = {
            "forward_not_pass": self.test_result_cnt['forward_fail_num'],
            "backward_not_pass": self.test_result_cnt['backward_fail_num'],
            "forward_and_backward_not_pass": self.test_result_cnt['forward_and_backward_fail_num'],
            "pass": self.test_result_cnt['success_num']
        }
        tb = PrettyTable()
        tb.add_column("Category", list(res_dict.keys()))
        tb.add_column("statistics", list(res_dict.values()))
        info_tb = str(tb)
        print_info_log(info_tb)

    def write_compare_csv(self):
        self.write_summary_csv()
        self.write_detail_csv()

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

    def write_detail_csv(self):
        test_rows = [[
            "Subject", "Cosine Similarity", "Cosine Similarity Pass", "Cosine Similarity Message",
            "Max Rel Error", "Max Rel Err Pass", "Max Rel Err Message",
            "Thousandth Rel Error Ratio", "Thousandth Rel Error Ratio Pass", "Thousandth Rel Error Ratio Message",
            "Ten Thousandth Rel Error Ratio", "Ten Thousandth Rel Error Ratio Pass", "Ten Thousandth Rel Error Ratio Message",
            "Compare Builtin Type", "Builtin Type Pass",
            "Builtin Type Message"
        ]]  
        for test_result in self.test_results:
            subject_prefix = test_result[0]
            fwd_result = test_result[3]
            bwd_result = test_result[4]
            if isinstance(fwd_result, list):
                for i, test_subject in enumerate(fwd_result):
                    subject = subject_prefix + ".forward.output." + str(i)
                    test_rows.append([subject] + list(test_subject))
            if isinstance(bwd_result, list):
                for i, test_subject in enumerate(bwd_result):
                    subject = subject_prefix + ".backward.output." + str(i)
                    test_rows.append([subject] + list(test_subject))

        write_csv(test_rows, self.detail_save_path)

    def record_results(self, *args):
        self.test_results.append(args)

    def register_compare_algorithm(self, name, compare_func, standard):
        self.compare_alg.update({name: (compare_func, standard)})

    def compare_output(self, api_name, bench_out, npu_out, bench_grad=None, npu_grad=None):
        if "dropout" in api_name:
            is_fwd_success, fwd_compare_alg_results = self._compare_dropout(bench_out, npu_out)    
        else:
            is_fwd_success, fwd_compare_alg_results = self._compare_core_wrapper(bench_out, npu_out)
        if bench_grad and npu_grad:
            if "dropout" in api_name:
                is_bwd_success, bwd_compare_alg_results = self._compare_dropout(bench_grad[0], npu_grad[0])
            else:
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
        detailed_result_total = []
        test_success_total = True
        for name in self.compare_alg.keys():
            alg = self.compare_alg[name][0]
            detailed_result, test_success = compare_core(bench_out, npu_out, alg)
            if name not in ["Max Relative Error", "Thousandth Relative Error Ratio", "Ten Thousandth Relative Error Ratio"]:
                test_success_total = test_success_total and test_success
            if detailed_result_total:
                for i in range(len(detailed_result_total)):
                    detailed_result_total[i] += detailed_result[i]
            else:
                detailed_result_total = detailed_result
        return test_success_total, detailed_result_total
    
    @staticmethod
    def _compare_dropout(bench_out, npu_out):
        tensor_num = bench_out.numel()
        if tensor_num >= 100:
            if abs((bench_out == 0).sum() - (npu_out == 0).cpu().sum()) / tensor_num < 0.1:
                return True, 1
            else:
                return False, 0
        else:
            return True, 1
