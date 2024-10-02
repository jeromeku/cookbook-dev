#!/bin/bash

#SBATCH --job-name=pt2pt-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:2                 
#SBATCH --time=0:30:00               
#SBATCH --partition=gpu_batch
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

ALGO=${1:-"all-gather"}
# Initialize a variable to track if --scan was passed
SCAN=$2

echo "START TIME: $(date)"

# auto-fail on any errors in this script
set -eo pipefail

# logging script's variables/commands for future debug needs
# set -x

LOG_PATH="main_log.txt"

echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE SLURM_NNODES=$SLURM_NNODES SLURM_JOBNAME=$SLURM_JOB_NAME"

GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
NNODES=$SLURM_NNODES

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000


LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role \$(hostname -s|tr -dc '0-9'): \
    --tee 3 \
    "
# max-size 2 ** 32 ~ 4GB
TRIALS=20
WARMUP=5
NELEMENTS=30 # 2 ** 30 ~ 1GB
DTYPE=uint8
PROGRAM="/home/jeromeku/savanna/cookbook-dev/benchmarks/communication/run_all.py"

PROGRAM_ARGS="--$ALGO --trials $TRIALS --warmup $WARMUP --elements-per-gpu $NELEMENTS --dtype $DTYPE $SCAN"
export CMD="$LAUNCHER $PROGRAM $PROGRAM_ARGS"

echo $CMD


# export NCCL_DEBUG=INFO

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
CONTAINER=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID"

# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS bash -c "$CMD" # 2>&1 | tee $LOG_PATH

echo "END TIME: $(date)"