import sys
import pandas as pd
from collections import defaultdict
import parser_helper


class NpuProfilingParser:
    def __init__(self, npu_step_time, npu_file_path):
        self.npu_json_file = npu_file_path.get('trace_view')
        self.npu_summary_file = npu_file_path.get('op_summary')
        self.npu_mem_file = npu_file_path.get('memory_record')
        self.profiling_info = parser_helper.ProfilingInfo()
        self.npu_step_time = npu_step_time
        self.parallel_time = 0
        self.aicore_time = 0

    def parse_npu_json_events(self):
        if not self.npu_json_file:
            print('Npu trace json file is not available.')
            return
        compute_time = 0
        min_ts = sys.float_info.max
        max_ts = sys.float_info.min
        data = parser_helper.read_json_file(self.npu_json_file)
        event_wait_sqe = defaultdict(list)
        ai_core_dict = defaultdict(list)
        event_wait_sqe_res = defaultdict(float)
        for dic in data:
            self.get_ts_by_task_type(dic, event_wait_sqe, ai_core_dict, event_wait_sqe_res)
            if ('name' in dic) and (dic.get('name') == 'compute_time'):
                compute_time += dic.get('dur')
                ts = dic.get('ts')
                min_ts = ts if ts < min_ts else min_ts
                max_ts = ts if ts > max_ts else max_ts
        # AI_CORE和EVENT_WAIT_SQE共存为计算流
        compute_stream = []
        parallel_stream = []
        # 不存在算子并行的情况
        if len(ai_core_dict) == 1:
            compute_stream.appen(min(ai_core_dict.keys()))
        elif len(ai_core_dict) == 2:  # 2个ai_core，存在并行流（当前最多2条算子计算流）
            compute_stream = list(event_wait_sqe.keys() & ai_core_dict.keys())
            parallel_stream = list(ai_core_dict.keys() - set(compute_stream))
        cs_event_wait_sqe_list = event_wait_sqe[compute_stream[0]]
        if parallel_stream:
            cs_ai_core_list = ai_core_dict[parallel_stream[0]]
            sorted(cs_event_wait_sqe_list, key=lambda x: (x[0]))
            sorted(cs_ai_core_list, key=lambda x: (x[0]))
            self.parallel_time = self.interval_intersection(cs_event_wait_sqe_list, cs_ai_core_list)
        self.profiling.compute_time = compute_time / 10 ** 6
        self.profiling_info.e2e_time = (max_ts - min_ts) / 1000 / 1000
        self.profiling_info.communication_not_overlapped = (event_wait_sqe_res[compute_stream[0]] - 
            self.parallel_time) / 10 ** 6
        time_required = (self.profiling_info.cube_time + self.profiling_info.vector_time) + \
            self.profiling_info.communication_not_overlapped
        if self.npu_step_time:
            self.profiling_info.scheduling_time = self.npu_step_time - time_required
        else:
            self.profiling_info.scheduling_time = self.profiling_info.e2e_time - time_required
        self.profiling_info.scheduling_ratio = self.profiling_info.scheduling_time / self.profiling_info.e2e_time

    def parse_npu_csv_events(self):
        if not self.npu_summary_file:
            print('Npu op summary csv file is not available.')
            return
        info = pd.read_csv(self.npu_summary_file, index_col=None)
        cube_time = 0.0
        vec_time = 0.0
        ai_core_time = 0.0
        vec_mac_flag = True  # True标记当前summary文件中存在pmu信息
        if info.get('aic_mac_time(us)') is None or info.get('aiv_vec_time(us)') is None:
            print('当前的profiling结果可能是极简模式,无法区分cube和vector,总的ai_core耗时会展示在vector算子列')
            vec_mac_flag = False
        for i in range(len(info['Model ID'])):
            task_type = info.loc[i, 'Task Type']
            if task_type not in ['AI_CORE']:
                continue
            task_durations = info.loc[i, 'Task Duration(us)']
            ai_core_time += task_durations
            if not vec_mac_flag:
                continue
            aiv_vec_time = info.loc[i, 'aiv_vec_time(us)']
            if aiv_vec_time > 0:
                vec_time += task_durations
        
        if vec_mac_flag:
            cube_time = (ai_core_time - vec_time) / 10 ** 6
            vec_time /= 10 ** 6
        else:
            vec_time = ai_core_time / 10 ** 6
        self.profiling_info.cube_time = cube_time
        self.profiling_info.vector_time = vec_time
        if not self.npu_mem_file:
            print('Npu op memory csv file is not available.')
            return
        try:
            info = pd.read_csv(self.npu_mem_file, usecols=['Total Reserved(MB)'], index_col=None)
        except ValueError:
            print('Npu profiling data does not contain memory info.')
        else:
            self.profiling_info.memory_used = max(info.get('Total Reserved(MB)')) / 1024

    @staticmethod
    def interval_intersection(cs_event_wait_sqe_list, cs_ai_core_list):
        ans = 0
        i = 0
        j = 0
        while i < len(cs_event_wait_sqe_list) and j < len(cs_ai_core_list):
            lo = max(cs_event_wait_sqe_list[i][0], cs_ai_core_list[j][0])
            hi = max(cs_event_wait_sqe_list[i][1], cs_ai_core_list[j][1])
            if lo < hi:
                ans += (hi - lo)
            if cs_event_wait_sqe_list[i][1] < cs_ai_core_list[j][1]:
                i += 1
            else:
                j += 1
        return ans


    @staticmethod
    def get_ts_by_task_type(dic, event_wait_sqe, ai_core_dict, res):
        if not dic.get('args'):
            return
        args = dic.get('args')
        if args.get('Stream Id'):
            stream_id = args.get('Stream Id')
            ts = dic.get('ts')
            dur = dic.get('dur')
            if args.get('Task Type') == 'EVENT_WAIT_SQE':
                res[stream_id] += dur
                event_wait_sqe[stream_id].append([ts, ts + dur])
            elif:
                ai_core_dict[stream_id].append([ts, ts + dur])