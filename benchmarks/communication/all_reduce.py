import os
import sys

import torch

COMMS_BENCH_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(COMMS_BENCH_DIR)

import communication.constants as COMM_CONST
from communication.utils import (
    benchmark_parser,
    bytes_to_human_readable,
    get_bw,
    get_metric_strings,
    get_scan_range,
    init_processes,
    print_header,
    print_rank_0,
    setup_single_payload,
    sync_all,
)


def timed_all_reduce(input, start_event, end_event, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        dist.all_reduce(input, async_op=args.async_op)
    sync_all()

    # time the actual comm op trials times and average it
    start_event.record()
    for i in range(args.trials):
        dist.all_reduce(input, async_op=args.async_op)
    end_event.record()
    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    tput, busbw = get_bw('all_reduce', size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    if not args.raw:
        size = bytes_to_human_readable(size)

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")


def run_all_reduce(args):
    # Prepare benchmark header
    print_header(args, 'all_reduce')

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if args.scan:
        payloads = get_scan_range(args)
        # loop over various tensor sizes
        for payload in payloads:
            input, _ = setup_single_payload(
                args, elements_per_gpu=payload, op="all_reduce"
            )
            timed_all_reduce(input, start_event, end_event, args)
    else:
        elements_per_gpu = 2 ** int(args.elements_per_gpu)
        input, _ = setup_single_payload(args, elements_per_gpu=elements_per_gpu, op="all_reduce")
        timed_all_reduce(input, start_event, end_event, args)


if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    rank = args.local_rank
    init_processes(local_rank=rank, args=args)
    run_all_reduce(args)
