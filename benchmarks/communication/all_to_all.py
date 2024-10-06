import os
import sys

import torch

COMMS_BENCH_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(COMMS_BENCH_DIR)

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


def timed_all_to_all(input, output, start_event, end_event, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        dist.all_to_all_single(output, input, async_op=args.async_op)
    sync_all()

    # time the actual comm op trials times and average it
    start_event.record()
    for i in range(args.trials):
        dist.all_to_all_single(output, input, async_op=args.async_op)
    end_event.record()
    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    tput, busbw = get_bw('all_to_all', size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    if not args.raw:
        size = bytes_to_human_readable(size)

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")


def run_all_to_all(args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    # Prepare benchmark header
    print_header(args, 'all_to_all')

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if args.scan:
        payloads = get_scan_range(args)
        for payload in payloads:
            input, output = setup_single_payload(
                args, elements_per_gpu=payload, op="all_to_all"
            )
            timed_all_to_all(input, output, start_event, end_event, args)
    else:
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        elements_per_gpu = 2 ** int(args.elements_per_gpu)
        input, output = setup_single_payload(args, elements_per_gpu=elements_per_gpu, op="all_to_all")

        if args.debug:
            for i in range(world_size):
                if i == global_rank:
                    print(f"Before AllToAll Input List at rank {global_rank}: {input}")
                dist.barrier()

        timed_all_to_all(input, output, start_event, end_event, args)

        if args.debug:
            for i in range(world_size):
                if i == global_rank:
                    print(f"AllToAll Results at rank {global_rank}: {output}")
                dist.barrier()


if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    rank = args.local_rank
    init_processes(local_rank=rank, args=args)
    run_all_to_all(args)
