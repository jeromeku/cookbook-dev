#!/bin/bash

#SBATCH --job-name=all-gather-gdb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:4                 
#SBATCH --time=0:30:00               
#SBATCH --account=dir_arc
#SBATCH --partition=pool0
#SBATCH --output=logs/sbatch_%N-%J.out
#SBATCH --error=logs/sbatch_%N-%J.err

echo "START TIME: $(date)"

# auto-fail on any errors in this script
set -eo pipefail

# logging script's variables/commands for future debug needs
# set -x

echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE SLURM_NNODES=$SLURM_NNODES"

GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
NNODES=$SLURM_NNODES

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
ROLE=$(hostname -s|tr -dc '0-9')
echo "MASTER_ADDR: $MASTER_ADDR MASTER_PORT: $MASTER_PORT ROLE: $ROLE"

LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0"
    # --role \$(hostname -s|tr -dc '0-9'): \
    # --tee 3 \

ROOT_DIR=`pwd`
PROGRAM="$ROOT_DIR/scripts/all-gather-gdb.py"
export CMD="$LAUNCHER $PROGRAM"

echo $CMD

CONTAINER="/scratch/fsw/portfolios/dir/projects/dir_arc/containers/clara-discovery+savanna+arc-evo2_efa+nv-latest-cascade-1.5.sqsh"

# Call from repo root so that ROOT_DIR points to repo root
SRUN_ARGS="--container-image $CONTAINER \
--container-workdir $ROOT_DIR \
--container-mounts $ROOT_DIR:$ROOT_DIR \
--output $ROOT_DIR/logs/srun_%N-%J.out \
--error $ROOT_DIR/logs/srun_%N-%J.err"

# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS bash -c "$CMD" 
# 2>&1 | tee $LOG_PATH

echo "END TIME: $(date)"


# max-size 2 ** 32 ~ 4GB
# TRIALS=5
# WARMUP=2
# MAXSIZE=31
# DTYPE=uint8

# export NCCL_DEBUG=INFO

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
# CONTAINER=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# SRUN_ARGS=" \
#     --wait=60 \
#     --kill-on-bad-exit=1 \
#     --jobid $SLURM_JOB_ID"