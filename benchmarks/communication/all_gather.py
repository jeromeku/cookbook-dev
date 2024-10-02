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
    setup_single_payload_all_gather,
    sync_all,
)


# Run all_gather and print metrics
def timed_all_gather(input, output, start_event, end_event, args):
    if args.dist == "torch":
        import torch.distributed as dist
    elif args.dist == "deepspeed":
        import deepspeed.comm as dist

    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        if args.dist == "torch":
            dist.all_gather_into_tensor(output, input)
        elif args.dist == "deepspeed":
            dist.allgather_fn(output, input, group=None, async_op=args.async_op)
    sync_all()

    # time the actual comm op trials times and average it
    start_event.record()
    for i in range(args.trials):
        if args.dist == "torch":
            dist.all_gather_into_tensor(output, input)
        elif args.dist == "deepspeed":
            dist.allgather_fn(output, input, group=None, async_op=args.async_op)
    end_event.record()
    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    tput, busbw = get_bw("all_gather", size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(
        args, tput, busbw, avg_duration
    )
    desc = f"{input.nelement()}x{input.element_size()}"

    if not args.raw:
        size = bytes_to_human_readable(size)

    print_rank_0(
        f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}"
    )


def run_all_gather(args):

    print_header(args, "all_gather")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if args.scan:
        # Create list of message sizes
        payloads = get_scan_range(args)
        # loop over various tensor sizes
        for payload in payloads:
            input, output = setup_single_payload_all_gather(
                args, elements_per_gpu=payload
            )
            timed_all_gather(input, output, start_event, end_event, args)
    else:
        elements_per_gpu = int(args.elements_per_gpu * COMM_CONST.ELEMENT_UNITS)
        input, output = setup_single_payload_all_gather(args, elements_per_gpu=elements_per_gpu)
        timed_all_gather(input, output, start_event, end_event, args)


if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    rank = args.local_rank
    init_processes(local_rank=rank, args=args)
    run_all_gather(args)
