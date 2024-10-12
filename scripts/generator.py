import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
def create_sbatch_script(args):
    # Set default values for SLURM parameters if not provided
    ntasks_per_node = args.ntasks_per_node
    num_gpus = args.num_gpus 
    num_nodes = args.num_nodes
    time = args.time or "0:30:00"
    account = args.account
    partition = args.partition 
    program = args.program
    master_port = args.master_port or 6000
    job_name = Path(program).stem
    output = args.output or f"logs/{job_name}/{partition}/n{num_nodes}-g{num_gpus}/sbatch.out"
    error = args.error or f"logs/{job_name}/{partition}/n{num_nodes}-g{num_gpus}/sbatch.err"

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --account={account}

echo "START TIME: $(date)"

set -eo pipefail

echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE SLURM_NNODES=$SLURM_NNODES"

GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
NNODES=$SLURM_NNODES

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

CMD="torchrun --nproc_per_node {num_gpus} \
--nnodes {num_nodes} \
--node_rank \$SLURM_PROCID \
--rdzv_endpoint $MASTER_ADDR:{master_port} \
--rdzv_backend c10d \
--max_restarts 0 {program}"

SRUN_ARGS="--output logs/{job_name}/{partition}/n{num_nodes}-g{num_gpus}/srun-%N.out \
--error logs/{job_name}/{partition}/n{num_nodes}-g{num_gpus}/srun-%N.err \
--container-image {args.container} \
--container-mounts {ROOT_DIR.as_posix()}:{ROOT_DIR.as_posix()}"
echo "Running: $CMD"
srun $SRUN_ARGS bash -c "$CMD"

echo "END TIME: $(date)"
"""

    # Save the script to a file
    script_filename = f"{job_name}-{partition}-n{num_nodes}-g{num_gpus}.sbatch"
    with open(script_filename, 'w') as f:
        f.write(sbatch_script)

    # Make the script executable
    os.chmod(script_filename, 0o755)
    print(f"SLURM script generated: {os.path.realpath(script_filename)}")


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM batch script.")
    parser.add_argument("--job-name", type=str,default="all-gather-bench", help="Job name for SLURM")
    parser.add_argument("--ntasks-per-node", default=1, type=int, help="Number of tasks per node")
    parser.add_argument("--time", type=str, default="0:30:00", help="Time for the job")
    parser.add_argument("--account", type=str, default="dir_arc", help="Account for SLURM")
    parser.add_argument("--partition", type=str, default="pool0_datahall_a", help="Partition for SLURM")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--error", type=str, help="Error file path")
    parser.add_argument("--container", type=str, help="Container image", default="/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/nvidia_evo2_efa_latest.sqsh")
    parser.add_argument("--program", type=Path, help="Program to run", default=SCRIPT_DIR / "all-gather-gdb.py")
    parser.add_argument("--master-port", type=int, default=6000, help="Master port for distributed training")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to allocate")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to allocate")

    args = parser.parse_args()
    create_sbatch_script(args)


if __name__ == "__main__":
    main()
