#!/bin/bash
#SBATCH -A m4229                # NERSC project allocation
#SBATCH -C gpu                  # GPU nodes (Perlmutter)
#SBATCH -q regular              # Regular queue for long jobs
#SBATCH -t 47:59:00             # Up to max 48 hrs allowed
#SBATCH -N 1                    # One node is typical for 4 GPUs
#SBATCH --ntasks-per-node=1     # One task per node for multi-GPU torch
#SBATCH -c 32                   # 32 CPUs for data loading etc.
#SBATCH --gpus-per-task=4       # 4 GPUs per task
#SBATCH --gpu-bind=map_gpu:0,1,2,3  # Bind 4 GPUs explicitly (optional)
#SBATCH --job-name=cgschnet_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-user=YOUR_EMAIL@lbl.gov
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=32
export TQDM_MININTERVAL=30
export TQDM_BAR_FORMAT="{desc}: {percentage:3.0f}% ({n_fmt}/{total_fmt}) ({elapsed}<{remaining}, {rate_fmt}{postfix})"

module load cudatoolkit
module load python

source ~/.bashrc
conda activate cgnet

cd ~/cgschnet/cgschnet/

DATASET=$SCRATCH/all_prots_aa
MODEL=$SCRATCH/models/bad_majewski

CMD="python scripts/train.py $DATASET $MODEL --epochs=100 --batch=8 --subsetpdbs=$SCRATCH/homeodomain_chignolin_trpcage10percent.txt --gpus=0,1,2,3 --apc=140000 --config=configs/config_john.yaml --wd=0 --lr=0.002 --exp-lr=0.90 --checkpoint-save=1 --mini-epoch=75"

echo "Running command: $CMD"
srun $CMD

