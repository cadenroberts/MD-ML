#!/bin/bash
#SBATCH -A m4229
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --mail-user=dsabo@ucsc.edu
#SBATCH --mail-type=ALL
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3
#SBATCH --job-name=daniel_openmm
#SBATCH --array=0

# nodes mean number of nodes NODES NODELIST(REASON) ex  4 gpub[020,030,035,038]
# -N is the number of nodes to run on
# -n is the number of tasks to run per node
# -c is CPUs per task

eval "$(~/bin/micromamba shell hook -s posix)"
micromamba activate openmm

#export OMP_NUM_THREADS=2  # if code is not multithreaded, otherwise set to 8 or 16
export OMP_NUM_THREADS=32
export DATA_DIR_PATH="$PSCRATCH/openmm_2023.12.20/"
echo "TASK_ID", $SLURM_ARRAY_TASK_ID
echo "BATCH_IDs:"
let TASK_A="$SLURM_ARRAY_TASK_ID * 4 + 0" ; echo TASK_A=$TASK_A
let TASK_B="$SLURM_ARRAY_TASK_ID * 4 + 1" ; echo TASK_B=$TASK_B
let TASK_C="$SLURM_ARRAY_TASK_ID * 4 + 2" ; echo TASK_C=$TASK_C
let TASK_D="$SLURM_ARRAY_TASK_ID * 4 + 3" ; echo TASK_D=$TASK_D

# Running multiple tasks per job: https://stackoverflow.com/questions/39186698/what-does-the-ntasks-or-n-tasks-does-in-slurm
srun -n 1 -c 32 -G 1 python3 batch_generate.py --batch-index $TASK_A --batch-size 20 --pool-size 10 --data-dir "$DATA_DIR_PATH" &
srun -n 1 -c 32 -G 1 python3 batch_generate.py --batch-index $TASK_B --batch-size 20 --pool-size 10 --data-dir "$DATA_DIR_PATH" &
srun -n 1 -c 32 -G 1 python3 batch_generate.py --batch-index $TASK_C --batch-size 20 --pool-size 10 --data-dir "$DATA_DIR_PATH" &
srun -n 1 -c 32 -G 1 python3 batch_generate.py --batch-index $TASK_D --batch-size 20 --pool-size 10 --data-dir "$DATA_DIR_PATH" &
wait
