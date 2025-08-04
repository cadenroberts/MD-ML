#!/usr/bin/env bash
#SBATCH --account=bbpa-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --mail-user=acbruce@ucsc.edu
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --job-name=andy_training


source /u/acbruce/.bashrc

micromamba activate cgschnet3

export OMP_NUM_THREADS=32

srun generate_ProtT5_embeddings.py \
     /work/hdd/bbpa/acbruce/training_data/benchmark-all-t5/cg_all_benchmark_finetuneE01/
