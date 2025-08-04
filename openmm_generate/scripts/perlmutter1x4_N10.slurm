#!/bin/bash
#SBATCH -A m4229
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --mail-user=dsabo@ucsc.edu
#SBATCH --mail-type=ALL
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=4
#SBATCH --job-name=daniel_openmm
#SBATCH --array=1-12

# nodes mean number of nodes NODES NODELIST(REASON) ex  4 gpub[020,030,035,038]
# -N is the number of nodes to run on
# -n is the number of tasks to run per node
# -c is CPUs per task

# Removed the "gpu-bind=map_gpu:0,1,2,3" parameter because we're going to map
# the GPUs within the python script.
# Also changed tasks per node to 1 and cores per task to 128 per
# https://docs.nersc.gov/systems/perlmutter/architecture/

eval "$(~/bin/micromamba shell hook -s posix)"
micromamba activate openmm

#export OMP_NUM_THREADS=2  # if code is not multithreaded, otherwise set to 8 or 16
export OMP_NUM_THREADS=32
export DATA_DIR_PATH="$PSCRATCH/openmm_2023.12.25/"
echo "TASK_ID", $SLURM_ARRAY_TASK_ID

# Because we're mapping the GPUs to subprocesses the pool size needs to be a multiple of the number of GPUs
srun -n 1 -c 128 -G 4 python3 batch_generate.py --batch-index $SLURM_ARRAY_TASK_ID --batch-size 40 --pool-size 8 --gpus="0,1,2,3" \
     --steps 100000 --report-steps 10 --data-dir "$DATA_DIR_PATH"
