#!/bin/bash
#SBATCH -A bdzk-delta-gpu
#SBATCH --partition=gpuA40x4 # run on A40s because they're cheaper than the A100s
#SBATCH -t 47:59:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH --gpus-per-task=4
#SBATCH --job-name=train-bad-majewski
#SBATCH --output=train_%j.out
#SBATCH --output="/work/hdd/bbpa/kbachelor/output_files/train-bad-majewski.stdout"
#SBATCH --error="/work/hdd/bbpa/kbachelor/output_files/train-bad-majewski.stderr"


# nodes mean number of nodes NODES NODELIST(REASON) ex  4 gpub[020,030,035,038]
# -N is the number of nodes to run on
# -n is the number of tasks to run per node
# -c is CPUs per task

#FIXME: How many threads does train.py need? Also change the -c above
export OMP_NUM_THREADS=64

export TQDM_MININTERVAL=30
export TQDM_BAR_FORMAT="{desc}: {percentage:3.0f}% ({n_fmt}/{total_fmt}) ({elapsed}<{remaining}, {rate_fmt}{postfix})"

# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions


# >>> mamba initialize >>>
# !! Contents within this block are managed by 'micromamba shell init' !!
export MAMBA_EXE='/u/kbachelor/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='y';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<

micromamba activate kevinschnet
cd /u/kbachelor/code/cgschnet/cgschnet/scripts/
# Train command:
#d=all_12368_cyrusc_081724
d=/work/hdd/bbpa/kbachelor/data/all_prots_aa
#m=all_12368_cyrusc_081724 #was the original model trained on the 4k dataset
#m=4k_v2 # same as the model below, but saved checkpoints every epoch
#m=4k_b5 # batch=5 didn't work on Delta, ran out of GPU memory
#m=6k_nnprior # 6k protein dataset with NN prior on bonds, angles and dihedrals
m=/work/hdd/bbpa/kbachelor/models/bad_majewski

# note that --batch=4 means that a batch has 4 frames for each protein. For 4k proteins => batch size=4*4k=16k (4 almost maxes out the GPU memory). For the 4k protein dataset, we have 225 batches.
cmd="python cgschnet/scripts/train.py $d $m --epochs=100 --batch=8 --subsetpdbs=/work/hdd/bbpa/kbachelor/homeodomain_chignolin_trpcage10percent.txt --gpus=0,1,2,3 --apc=140000 --config=cgschnet/configs/config_john.yaml --wd=0 --lr=0.002 --exp-lr=0.90 --checkpoint-save=1 --mini-epoch=75"
#cmd="python cgschnet/scripts/train.py data_generation/preprocessed_datasets/$d data_generation/models/$m --gpus=0,1,2,3 --apc=140000 --config=cgschnet/configs/config_cutoff2_seq6.yaml --wd=0 --lr=0.002 --exp-lr=0.90 --batch=4 --epochs=80 --checkpoint-save=1 --subsetpdbs=train_subset_ok.txt"
echo $cmd
srun $cmd
