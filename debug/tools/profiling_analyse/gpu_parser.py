from collections import defaultdict
import pandas as pd

import parser_helper


class GpuProfilingParser:
    def __init__(self, args):
        self.trace_events = self.read_profiling_json_file(args.gpu)
        self.one_step_time = args.gpu_step
        self.profiling_info = parser_helper.ProfilingInfo()

    @staticmethod
    def read_profiling_json_file(json_path):
        data = parser_helper.read_json_file(json_path)
        if 'traceEvents' not in data:
            raise RuntimeError("The gpu profiling json doesn't contain traceEvents data.")
        return data.get('traceEvents')

    def parse_events(self):
        cube_time = 0.0
        all_op_time = 0.0
        op_list = []
        compute_stream_dur = 0.0  # 计算流耗时
        marks = defaultdict(int) # mark for compute communication_not_overlapped time

        for event in self.trace_events:
            if not isinstance(event, dict):
                continue
            if event.get('args') and event.get('args').get('stream') == 7:
                compute_stream_dur += float(event.get('dur'))
            if not {'name', 'cat', 'dur', 'ts'} < event.keys():
                continue
            name = event.get('name')
            dur = event.get('dur')
            ts = event.get('ts')
            cat = event.get('cat')
            if cat.lower() != 'kernel':
                continue
            if 'nccl' in name:
                for timestep in range(ts + 1, ts + dur + 1):
                    marks[str(timestep)] += 1 # mark this timestep in communication stream
                continue
            else:
                for timestep in range(ts + 1, ts + dur + 1):
                    marks[str(timestep)] += -100   # mark this timestep in compute stream
            if 'gemm' in name:
                cube_time += float(dur)
            all_op_time += float(dur)
            op_list.append([ts, name, cat, dur])
        op_dataframe = pd.DataFrame(op_list, columns=['time start', 'name', 'cat', 'dur'])
        op_dataframe.to_csv('gpu_perf.csv', index=False)
        self.profiling_info.communication_not_overlapped = len([_ for _,value in marks.items() if value > 0]) / 10 ** 6
        self.profiling_info.cube_time = cube_time / 10 ** 6
        self.profiling_info.vector_time = (all_op_time - cube_time) / 10 ** 6
        self.parse_e2e_time()
        if self.one_step_time:
            self.profiling_info.scheduling_time = self.one_step_time - all_op_time / 10 ** 6 - \
                                                  self.profiling_info.communication_not_overlapped
        else:
            self.profiling_info.scheduling_time = self.profiling_info.e2e_time - all_op_time / 10 ** 6 - \
                                                  self.profiling_info.communication_not_overlapped
        self.profiling_info.scheduling_ratio = self.profiling_info.scheduling_time / self.profiling_info.e2e_time
        self.parse_memory_reserved()

    def parse_e2e_time(self):
        timeline = sorted(self.trace_events, key=lambda event: event.get('ts'))
        self.profiling_info.e2e_time = (timeline[-1].get('ts') - timeline[0].get('ts')) / 10 ** 6

    def parse_memory_reserved(self):
        memories = [event.get('args').get('Total Reserved') for event in self.trace_events if
                    event.get('name') == '[memory]' and event.get('args').get('Device Id') >= 0]
        if not memories:
            print("Gpu profiling data doesn't contain memory info")
            return
        self.profiling_info.memory_used = max(memories) / 1024 ** 3
