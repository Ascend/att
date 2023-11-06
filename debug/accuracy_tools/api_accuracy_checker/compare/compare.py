# 进行比对及结果展示
import os
import time
from rich.table import Table
from rich.console import Console
from api_accuracy_checker.compare.algorithm import compare_core, cosine_sim, cosine_standard, get_max_rel_err, get_max_abs_err, \
    compare_builtin_type, get_rel_err_ratio_hundredth, get_rel_err_ratio_thousandth, get_rel_err_ratio_ten_thousandth
from api_accuracy_checker.common.utils import get_json_contents, print_info_log, print_error_log, write_csv, CompareException

from api_accuracy_checker.compare.compare_utils import CompareConst 
from api_accuracy_checker.common.config import msCheckerConfig


class Comparator:
    TEST_FILE_NAME = "accuracy_checking_result_" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    DETAIL_TEST_FILE_NAME = "accuracy_checking_details_" + time.strftime("%Y%m%d%H%M%S") + ".csv"

    # consts for result csv 
    COLUMN_API_NAME = "API name"
    COLUMN_FORWARD_SUCCESS = "Forward Test Success"
    COLUMN_BACKWARD_SUCCESS = "Backward Test Success"
    COLUMN_STACK_INFO = "Traceback callstack info"

    def __init__(self, result_save_path, stack_info_json_path=None):
        self.save_path = os.path.join(result_save_path, self.TEST_FILE_NAME)
        if os.path.exists(self.save_path):
            raise ValueError(f"file {self.save_path} already exists, please remove it first or use a new dump path")
        self.detail_save_path = os.path.join(result_save_path, self.DETAIL_TEST_FILE_NAME)
        if os.path.exists(self.detail_save_path):
            raise ValueError(f"file {self.detail_save_path} already exists, please remove it first or use a new dump path")
        if stack_info_json_path:
            self.stack_info = get_json_contents(stack_info_json_path)
        else:
            self.stack_info = None
        self.compare_alg = {}
        self.register_compare_algorithm("Cosine Similarity", cosine_sim, cosine_standard)
        self.register_compare_algorithm("Max Relative Error", get_max_rel_err, None)
        self.register_compare_algorithm("Max Absolute Error", get_max_abs_err, None)
        self.register_compare_algorithm("Hundredth Relative Error Ratio", get_rel_err_ratio_hundredth, None)
        self.register_compare_algorithm("Thousandth Relative Error Ratio", get_rel_err_ratio_thousandth, None)
        self.register_compare_algorithm("Ten Thousandth Relative Error Ratio", get_rel_err_ratio_ten_thousandth, None)
        self.register_compare_algorithm("Default: isEqual", compare_builtin_type, None)

        self.test_result_cnt = {
            "forward_fail_num": 0, "backward_fail_num": 0, "forward_and_backward_fail_num": 0, "success_num": 0,
            "total_num": 0, "forward_or_backward_fail_num": 0
        }
        self.result_save_path = result_save_path
        self.write_csv_title()

    def print_pretest_result(self):
        if self.test_result_cnt.get("total_num") != 0:
            passing_rate = str(self.test_result_cnt.get("success_num") / self.test_result_cnt.get("total_num"))
        else:
            passing_rate = "0"

        console = Console()
        table_total = Table(
            show_header=True, title="Overall Statistics", show_lines=True, width=75
        )
        table_total.add_column("Result")
        table_total.add_column("Statistics")
        table_total.add_row("[green]Pass[/green]", str(self.test_result_cnt.get("success_num")))
        table_total.add_row("[red]Fail[/red]", str(self.test_result_cnt.get("forward_and_backward_fail_num") +
                                                   self.test_result_cnt.get("forward_or_backward_fail_num")))
        table_total.add_row("Passing Rate", passing_rate)

        table_detail = Table(
            show_header=True, title="Detail Statistics", show_lines=True, width=75
        )
        table_detail.add_column("Result")
        table_detail.add_column("Statistics")
        table_detail.add_row("Only Forward Fail", str(self.test_result_cnt.get("forward_fail_num")))
        table_detail.add_row("Only Backward Fail", str(self.test_result_cnt.get("backward_fail_num")))
        table_detail.add_row(
            "Both Forward & Backward Fail", str(self.test_result_cnt.get("forward_and_backward_fail_num")))

        console.print(table_total)
        console.print(table_detail)

    def write_csv_title(self):
        summary_test_rows = [[self.COLUMN_API_NAME, self.COLUMN_FORWARD_SUCCESS, self.COLUMN_BACKWARD_SUCCESS, "Message"]]
        write_csv(summary_test_rows, self.save_path)

        detail_test_rows = [[
            "Npu Name", "Bench Dtype", "NPU Dtype", "Shape",
            "Cosine Similarity", "Cosine Similarity Message",
            "Max Rel Error", "Max Rel Err Message",
            "Max Abs Error", "Max Abs Err Message",
            "Relative Error (hundredth)", "Relative Error (dual hundredth) Message",
            "Relative Error (dual thousandth)", "Relative Error (dual thousandth) Message",
            "Relative Error (dual ten thousandth)", "Relative Error (dual ten thousandth) Message",
            "Compare Builtin Type", "Builtin Type Message",
            "Pass"
        ]]  
        write_csv(detail_test_rows, self.detail_save_path)

    def write_summary_csv(self, test_result):
        test_rows = []
        if self.stack_info:
            test_rows[0].append(self.COLUMN_STACK_INFO)

        name = test_result[0]
        df_row = list(test_result[:3])
        if test_result[1] == "SKIP" or test_result[2] == "SKIP":
            df_row.append(test_result[3])
        if self.stack_info:
            stack_info = "\n".join(self.stack_info[name])
            df_row.append(stack_info)
        test_rows.append(df_row)
        write_csv(test_rows, self.save_path)

    def write_detail_csv(self, test_result):
        test_rows = []

        subject_prefix = test_result[0]
        fwd_result = test_result[3]
        bwd_result = test_result[4]
        if isinstance(fwd_result, list):
            for i, test_subject in enumerate(fwd_result):
                subject = subject_prefix + ".forward.output." + str(i)
                test_subject = ["{:.{}f}".format(item, msCheckerConfig.precision) if isinstance(item, float) else item for item in test_subject]
                test_rows.append([subject] + list(test_subject))
        if isinstance(bwd_result, list):
            for i, test_subject in enumerate(bwd_result):
                subject = subject_prefix + ".backward.output." + str(i)
                test_subject = ["{:.{}f}".format(item, msCheckerConfig.precision) if isinstance(item, float) else item for item in test_subject]
                test_rows.append([subject] + list(test_subject))

        write_csv(test_rows, self.detail_save_path)

    def record_results(self, *args):
        self.write_summary_csv(args)
        self.write_detail_csv(args)


    def register_compare_algorithm(self, name, compare_func, standard):
        self.compare_alg.update({name: (compare_func, standard)})

    def compare_output(self, api_name, bench_out, npu_out, bench_grad=None, npu_grad=None):
        self.test_result_cnt["total_num"] += 1
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
            is_bwd_success, bwd_compare_alg_results = False, None
        self.record_results(api_name, is_fwd_success, is_bwd_success, fwd_compare_alg_results, bwd_compare_alg_results)
        if is_fwd_success and is_bwd_success:
            self.test_result_cnt['success_num'] += 1
        elif not is_fwd_success and not is_bwd_success:
            self.test_result_cnt['forward_and_backward_fail_num'] += 1
        elif not is_fwd_success:
            self.test_result_cnt['forward_fail_num'] += 1
            self.test_result_cnt['forward_or_backward_fail_num'] += 1
        else:
            self.test_result_cnt['backward_fail_num'] += 1
            self.test_result_cnt['forward_or_backward_fail_num'] += 1
        return is_fwd_success, is_bwd_success


    def _compare_core_wrapper(self, bench_out, npu_out):
        detailed_result_total = []
        bench_dtype_total = []
        npu_dtype_total = []
        shape_total = []
        test_success_total = []
        max_abs_error_success = []
        cosine_success = []
        for name in self.compare_alg.keys():
            alg = self.compare_alg[name][0]
            detailed_result, test_success, bench_dtype, npu_dtype, shape = compare_core(bench_out, npu_out, alg)
            bench_dtype_total = bench_dtype
            npu_dtype_total = npu_dtype
            shape_total = shape
            if name not in ["Cosine Similarity", "Max Relative Error", "Max Absolute Error"]:
                test_success_total.append(test_success)
            if name == "Cosine Similarity":
                cosine_success.append(test_success)
            if name == "Max Relative Error":
                max_abs_error_success.append(test_success)
            if detailed_result_total:
                for i, detailed_result_item in enumerate(detailed_result):
                    detailed_result_total[i] += detailed_result_item
            else:
                detailed_result_total = detailed_result
        test_all_result = [CompareConst.PASS for _ in range(len(detailed_result_total))]
        for i, _ in enumerate(test_all_result):
            if not cosine_success[0][i] or CompareConst.ERROR == cosine_success[0][i]:
                test_all_result[i] = CompareConst.ERROR
            elif max_abs_error_success[0][i] or CompareConst.PASS == max_abs_error_success[0][i]:
                test_all_result[i] = CompareConst.PASS
            else:
                test_success_column = [test_success_single[i] for test_success_single in test_success_total]
                if CompareConst.ERROR in test_success_column or False in test_success_column:
                    test_all_result[i] = CompareConst.ERROR
                elif CompareConst.WARNING in test_success_column:
                    test_all_result[i] = CompareConst.WARNING
        # dtype加到所有指标的前面, 是否pass放到所有指标的后面
        try:
            for i, detailed_tuple in enumerate(detailed_result_total):
                detailed_result = list(detailed_tuple)
                detailed_result.insert(0, bench_dtype_total[i])
                detailed_result.insert(1, npu_dtype_total[i])
                detailed_result.insert(2, shape_total[i])
                detailed_result.append(test_all_result[i])
                detailed_result_total[i] = tuple(detailed_result)
        except IndexError as error:
            print_error_log(f"There is index error.\n{str(error)}")
            raise CompareException(CompareException.INVALID_DATA_ERROR) from error
        test_final_success = False if CompareConst.ERROR in test_all_result or CompareConst.WARNING in test_all_result else True
        return test_final_success, detailed_result_total
    
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
