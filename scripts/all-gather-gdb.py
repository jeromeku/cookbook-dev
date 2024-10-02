import gc
import os
import socket
import time
import warnings

import torch

SIZE = 20e9

warnings.filterwarnings('ignore', message='process group has NOT been destroyed before we destruct ProcessGroupNCCL.')
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
print(f"LOCAL_RANK: {local_rank}, RANK: {rank}, WORLD_SIZE: {world_size}")

torch.distributed.init_process_group(backend='nccl')#, init_method='env://')

data = torch.empty(int(SIZE / world_size), device=f'cuda:{local_rank}', dtype=torch.uint8)
tensor_list = torch.zeros((world_size,) + data.shape, device=data.device, dtype=data.dtype)

torch.distributed.barrier()
torch.cuda.synchronize()

gc.disable()

# timers
events = []
for i in range(20):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    events.append((start, stop))

for i in range(20):
    if i == 1:
        total_duration = time.time()

    start, stop = events[i]
    start.record()
    torch.cuda.nvtx.range_push(f'all_gather{i}')
    torch.distributed.all_gather_into_tensor(tensor_list, data)
    torch.cuda.nvtx.range_pop()
    stop.record()

torch.cuda.synchronize()

if rank == 0:
    print(f'world_size={world_size} bytes={data.numel()} total_duration={time.time()-total_duration}')
    for start, stop in events: print(f'xfer time (ms): {start.elapsed_time(stop)}')