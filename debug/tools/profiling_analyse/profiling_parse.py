import argparse
import os

from prettytable import PrettyTable

from gpu_parser import GpuProfilingParser
from npu_parser import NpuProfilingParser
from parser_helper import ProfilingInfo


def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', required=False, default='', metavar='(FILE)', help='Gpu profiling json file.')
    parser.add_argument('-n', '--npu', required=False, default='', metavar='(FILE)',
                        help='Npu single core profiling root path.')
    parser.add_argument('-ns', '--npu_step', required=False, default='', metavar='(FILE)', type=float, 
                        help='Npu one step time(s).')
    return parser.parse_args()


def show_table(gpu_profiling_info, npu_profiling_info):
    table = PrettyTable()
    table.title = '大模型性能拆解'
    table.field_names = ['', 'cube算子', 'vector算子', '大kernel算子', '通信', '调度耗时', '调度占比', '内存',
                         'E2E性能值']
    table.add_row(['GPU基线', f'{gpu_profiling_info.cube_time:.3f}s', f'{gpu_profiling_info.vector_time:.3f}s',
                  f'{gpu_profiling_info.large_kernel:.3f}s', f'{gpu_profiling_info.communication_not_overlapped: .3f}s',
                  f'{gpu_profiling_info.scheduling_time:.3f}', f'{gpu_profiling_info.scheduling_ratio:.2%}',
                  f'{gpu_profiling_info.memory_used:.2f}G', f'{gpu_profiling_info.e2e_time:.3f}s'])
    table.add_row(['当前现状', f'{npu_profiling_info.cube_time:.3f}s', f'{npu_profiling_info.vector_time:.3f}s',
                  f'{npu_profiling_info.large_kernel:.3f}s', f'{npu_profiling_info.communication_not_overlapped: .3f}s',
                  f'{npu_profiling_info.scheduling_time:.3f}', f'{npu_profiling_info.scheduling_ratio:.2%}',
                  f'{npu_profiling_info.memory_used:.2f}G', f'{npu_profiling_info.e2e_time:.3f}s'])
    print(table)


def parse_gpu(args):
    if args.gpu:
        gpu_parser = GpuProfilingParser(args.gpu)
        gpu_parser.parse_events()
        return gpu_parser.profiling_info
    print('Gpu trace json file is not specified.')
    return ProfilingInfo()


def parse_npu(args, npu_path):
    if not npu_path.get('trace_view'):
        print('Npu trace json file is not available.')
        return ProfilingInfo()
    if not npu_path.get('op_summary'):
        print('Npu op summary csv file is not available.')
        return ProfilingInfo()
    if not npu_path.get('memory_record'):
        print('Npu op memory csv file is not available.')
        return ProfilingInfo()
    npu_parser = NpuProfilingParser(args.npu_step, npu_path)
    npu_parser.parse_npu_csv_events()
    npu_parser.parse_npu_json_events()
    return npu_parser.profiling_info


def main():
    args = parse_command()
    npu_path = {'trace_view': None, 'memory_record': None, 'op_summary': None}
    for root, _, files in os.walk(args.npu):
        for file in files:
            if file == 'trace_view.json':
                npu_path['trace_view'] = os.path.join(root, file)
            if file == 'memory_record.csv':
                npu_path['memory_record'] = os.path.join(root, file)
            if 'op_summary' in file:
                npu_path['op_summary'] = os.path.join(root, file)
    show_table(parse_gpu(args), parse_npu(args, npu_path))


if __name__ == '__main__':
    main()
