#!/bin/bash

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --job-name) JOB_NAME="$2"; shift ;;
        --ntasks-per-node) NTASKS_PER_NODE="$2"; shift ;;
        --time) TIME="$2"; shift ;;
        --account) ACCOUNT="$2"; shift ;;
        --partition) PARTITION="$2"; shift ;;
        --output) OUTPUT="$2"; shift ;;
        --error) ERROR="$2"; shift ;;
        --container) CONTAINER="$2"; shift ;;
        --program) PROGRAM="$2"; shift ;;
        --master-port) MASTER_PORT="$2"; shift ;;
        --num-gpus) NUM_GPUS="$2"; shift ;;
        --num-nodes) NUM_NODES="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Default values if not provided
JOB_NAME=${JOB_NAME:-all-gather-bench}
NTASKS_PER_NODE=${NTASKS_PER_NODE:-1}
NUM_GPUS=${NUM_GPUS:-8}
NUM_NODES=${NUM_NODES:-1}
TIME=${TIME:-0:30:00}
ACCOUNT=${ACCOUNT:-dir_arc}
PARTITION=${PARTITION:-pool0_datahall_a}
OUTPUT=${OUTPUT:-logs/$JOB_NAME/$PARTITION/n$NUM_NODES-g$NUM_GPUS/sbatch.out}
ERROR=${ERROR:-logs/$JOB_NAME/$PARTITION/n$NUM_NODES-g$NUM_GPUS/sbatch.err}
CONTAINER=${CONTAINER:-"/scratch/fsw/portfolios/dir/projects/dir_arc/containers/clara-discovery+savanna+arc-evo2_efa+nv-latest-cascade-1.5.sqsh"}
PROGRAM=${PROGRAM:-"scripts/all-gather-gdb.py"}
ROOT_DIR=${ROOT_DIR:-$(pwd)}
MASTER_PORT=${MASTER_PORT:-6000}

# Generates SLURM batch directives
SBATCH="#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=$NTASKS_PER_NODE
#SBATCH --gres=gpu:$NUM_GPUS
#SBATCH --time=$TIME
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --output=$OUTPUT
#SBATCH --error=$ERROR

echo \"START TIME: \$(date)\"

set -eo pipefail

echo \"SLURM_GPUS_ON_NODE=\$SLURM_GPUS_ON_NODE SLURM_NNODES=\$SLURM_NNODES\"

GPUS_PER_NODE=\$SLURM_GPUS_ON_NODE
NNODES=\$SLURM_NNODES

MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
# ROLE=\${ROLE:-\$(hostname -s|tr -dc '0-9')}

CMD=\"torchrun --nproc_per_node $NUM_GPUS \
--nnodes $NUM_NODES \
--node_rank \\$SLURM_PROCID \
--rdzv_endpoint \$MASTER_ADDR:$MASTER_PORT \
--rdzv_backend c10d \
--max_restarts 0 $PROGRAM\"

SRUN_ARGS=\"--container-image $CONTAINER \
--container-workdir $ROOT_DIR \
--container-mounts $ROOT_DIR:$ROOT_DIR \
--output $ROOT_DIR/logs/$JOB_NAME/$PARTITION/n$NUM_NODES-g$NUM_GPUS/srun-%N.out \
--error $ROOT_DIR/logs/$JOB_NAME/$PARTITION/n$NUM_NODES-g$NUM_GPUS/srun-%N.err\"

echo \"Running: \$CMD\"
srun \$SRUN_ARGS bash -c \"\$CMD\"

echo \"END TIME: \$(date)\"
"

# Save the script to a file
OUTPUT_FILE=${JOB_NAME}-${PARTITION}-n${NUM_NODES}-g${NUM_GPUS}.sbatch
echo "$SBATCH" > $OUTPUT_FILE
script=$(realpath $OUTPUT_FILE)
chmod +x $script
echo "SLURM script generated: $script"
# sbatch $script