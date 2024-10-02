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


def timed_pt2pt(input, start_event, end_event, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        if dist.get_rank() == 0:
            if args.async_op:
                dist.isend(input, 1)
            else:
                dist.send(input, 1)
        if dist.get_rank() == 1:
            if args.async_op:
                dist.irecv(input, src=0)
            else:
                dist.recv(input, src=0)
    sync_all()

    # time the actual comm op trials times and average it
    start_event.record()
    for i in range(args.trials):
        if dist.get_rank() == 0:
            if args.async_op:
                dist.isend(input, 1)
            else:
                dist.send(input, 1)
        if dist.get_rank() == 1:
            if args.async_op:
                dist.irecv(input, src=0)
            else:
                dist.recv(input, src=0)

    end_event.record()
    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    tput, busbw = get_bw('pt2pt', size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    if not args.raw:
        size = bytes_to_human_readable(size)

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")


def run_pt2pt(args):

    print_header(args, 'pt2pt')
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if args.scan:
        payloads = get_scan_range(args)
        for payload in payloads:
            input, _ = setup_single_payload(
                args, elements_per_gpu=payload, op="pt2pt"
            )
            timed_pt2pt(input, start_event, end_event, args)
    else:
        elements_per_gpu = 2 ** int(args.elements_per_gpu)
        input, _ = setup_single_payload(args, elements_per_gpu=elements_per_gpu, op="pt2pt")
        timed_pt2pt(input, start_event, end_event, args)


if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    rank = args.local_rank
    init_processes(local_rank=rank, args=args)
    run_pt2pt(args)
