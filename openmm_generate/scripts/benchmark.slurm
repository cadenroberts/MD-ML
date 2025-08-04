#!/bin/bash
#SBATCH -A m4229
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --job-name=openmm_benchmark_v4
#SBATCH --array=0-7
#SBATCH --output=logs/benchmark_v4_%A_%a.out
#SBATCH --error=logs/benchmark_v4_%A_%a.err

# Setup micromamba if needed
eval "$(/global/homes/c/cawrober/micromamba/micromamba shell hook -s posix)"
micromamba activate openmm

export OMP_NUM_THREADS=32

export DATA_DIR_PATH="$PSCRATCH/benchmark_test_v4"

srun -n 1 -c 32 -G 1 python3 scripts/batch_generate.py scripts/benchmark_pdbs.json \
  --prepare \
  --remove-ligands \
  --batch-size 1 \
  --batch-index $SLURM_ARRAY_TASK_ID \
  --pool-size 1 \
  --steps 1000 \
  --report-steps 100 \
  --data-dir "$DATA_DIR_PATH"

