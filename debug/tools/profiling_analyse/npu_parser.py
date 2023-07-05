import sys
import pandas as pd
import parser_helper


class NpuProfilingParser:
    def __init__(self, npu_file_path):
        self.npu_json_file = npu_file_path.get('trace_view')
        self.npu_summary_file = npu_file_path.get('op_summary')
        self.npu_mem_file = npu_file_path.get('memory_record')
        self.profiling_info = parser_helper.ProfilingInfo()

    def parse_npu_json_events(self):
        conn_time = 0.0
        compute_time = 0.0
        min_ts = sys.float_info.max
        max_ts = sys.float_info.min
        data = parser_helper.read_json_file(self.npu_json_file)
        for dic in data:
            if dic.get('name') == 'communication_not_overlapped':
                conn_time += float(dic.get('dur'))
            if dic.get('name') == 'compute_time':
                compute_time += float(dic.get('dur'))
            if dic.get('ts'):
                ts = dic.get('ts')
                min_ts = ts if ts < min_ts else min_ts
                max_ts = ts if ts > max_ts else max_ts
        self.profiling_info.e2e_time = (max_ts - min_ts) / 10 ** 6
        self.profiling_info.communication_not_overlapped = conn_time / 10 ** 6
        compute_time = compute_time / 10 ** 6
        self.profiling_info.scheduling_time = self.profiling_info.e2e_time - compute_time
        self.profiling_info.scheduling_ratio = self.profiling_info.scheduling_time / self.profiling_info.e2e_time

    def parse_npu_csv_events(self):
        # 读取csv文件
        info = pd.read_csv(self.npu_summary_file, index_col=None)
        # 用一个字典保存各类算子的统计结果
        op_statics_result = {}
        # cube和vector时间
        cube_time = 0.0
        vec_time = 0.0
        length = len(info['Model ID'])
        if info.get('aic_mac_time(us)') is None or info.get('aiv_vec_time(us)') is None:
            raise ValueError('There is no cube time or vector time in the csv!The aic_mac_time(us) and '
                             'aiv_vec_time(us) are necessary for the determination of cube and vector.')

        for i in range(length):
            op_type = info.loc[i, 'OP Type']
            task_type = info.loc[i, 'Task Type']
            task_durations = info.loc[i, 'Task Duration(us)']
            aic_mac_time = info.loc[i, 'aic_mac_time(us)']
            aiv_vec_time = info.loc[i, 'aiv_vec_time(us)']

            if task_type in ['AI_CORE']:
                if aiv_vec_time > aic_mac_time:
                    vec_time += task_durations
                    if op_statics_result.get(op_type) is None:
                        op_statics_result[op_type] = [task_durations, 'vector']
                    else:
                        op_statics_result[op_type][0] += task_durations
                else:
                    cube_time += task_durations
                    if op_statics_result.get(op_type) is None:
                        op_statics_result[op_type] = [task_durations, 'cube']
                    else:
                        op_statics_result[op_type][0] += task_durations

        info = pd.read_csv(self.npu_mem_file, usecols=['Total Reserved(MB)'], index_col=None)
        self.profiling_info.memory_used = max(info.get('Total Reserved(MB)')) / 1024
        self.profiling_info.cube_time = cube_time / 10 ** 6
        self.profiling_info.vector_time = vec_time / 10 ** 6
