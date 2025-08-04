#!/usr/bin/env bash
#SBATCH --account=bbpa-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --mail-user=acbruce@ucsc.edu
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=4
#SBATCH --job-name=andy_training

source /u/acbruce/.bashrc

micromamba activate cgschnet3

export OMP_NUM_THREADS=32

srun python3 train.py \
     /work/hdd/bbpa/acbruce/training_data/benchmark-all-t5/cg_all_benchmark_finetuneE01/ \
     --gpus 0,1,2,3 \
     --atoms-per-call 140000 \
     --epoch 35 \
     --config=../configs/config_cutoff2_seq6_harBAD_exemb.yaml \
     --embedding protT5_embedding.npy \
     --wd=0 \
     --lr=0.001 \
     --exp-lr=0.85 \
     --batch=4
