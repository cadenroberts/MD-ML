#!/usr/bin/env bash
#SBATCH --account=bbpa-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=4
#SBATCH --job-name=preprocess_raz
#SBATCH --output=preprocess_%A_%a.out
#SBATCH --array=0-9  # Array job with 10 tasks (0 to 9)

export OMP_NUM_THREADS=64

#### CONFIG: Before running the script, set your environment to the correct conda environment and activate it below
source env_raz.sh
micromamba activate benchmark
#### END CONFIG

cd ../../

srun python cgschnet/scripts/preprocess.py data_generation/saved_datasets/all_12368_cyrusc_081724/new_data  --output=data_generation/preprocessed_datasets/6k_flex --prior=CA_lj_angleXCX_dihedralX_flex --resume --num-cores=4 --jobid=${SLURM_ARRAY_TASK_ID} --totalNrJobs=10
