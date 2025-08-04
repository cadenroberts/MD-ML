#!/bin/bash
#SBATCH -A m4229
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --mail-user=cawrober@ucsc.edu
#SBATCH --mail-type=ALL
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=4
#SBATCH --job-name=daniel_openmm
#SBATCH --array=0-2

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
export PDB_ID_LIST_FILE="openmm_2023.12.26_N100.json"
export DATA_DIR_PATH="$PSCRATCH/openmm_2023.12.26/"
echo "TASK_ID", $SLURM_ARRAY_TASK_ID

# Because we're mapping the GPUs to subprocesses the pool size needs to be a multiple of the number of GPUs
srun -n 1 -c 128 -G 4 python3 batch_generate.py "$PDB_ID_LIST_FILE" --batch-index $SLURM_ARRAY_TASK_ID --batch-size 32 --pool-size 4 --gpus="0,1,2,3" \
     --steps 1000000 --report-steps 100 --data-dir "$DATA_DIR_PATH"
