#!/usr/bin/env bash
#SBATCH --account=bbpa-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=4
#SBATCH --job-name=benchmark
#SBATCH --output=benchmark_%j.out

#### CONFIG: Before running the script, set these settings and un-comment one of the srun commands below. We have commands for: 1) standard benchmark, 2) benchmarking of all checkpoints and 3) benchmark of more exotic models, such as those with neural net priors

# Choose model to benchmark
m=all_12368_cyrusc_081724
#m=4k_v2
#m=6k_nnprior

# Set micromamba/conda environment load script  
source env_raz.sh # change to your own conda environment and activate it below 
micromamba activate benchmark # install from conda/andy_benchmark.yml
export OMP_NUM_THREADS=64

# Tip: first run the below commands (without srun) inside an interactive session to check that it works. 
##### END CONFIG #####

# STANDARD BENCHMARK, 6 PROTEINS
srun python3 ./gen_benchmark.py --temperature 300 --machine delta --use-cache ../../data_generation/models/$m  --proteins bba chignolin homeodomain proteinb trpcage wwdomain


##### NEURAL NET PRIOR ###########
#srun python3 ./gen_benchmark.py ../../data_generation/models/6k_nnprior  --temperature 300  --machine delta --prior-nn ../../data_generation/preprocessed_datasets/6k_flex  --proteins bba chignolin homeodomain proteinb trpcage wwdomain 

# srun python3 ./gen_benchmark.py ../../data_generation/models/$m  --temperature 300   --machine delta   --proteins bba chignolin homeodomain proteinb trpcage wwdomain --prior-only --prior-nn ../../data_generation/preprocessed_datasets/6k_flex

######## ALL CHECKPOINTS #########
#srun python3 ./gen_benchmark_allcheckpoints.py --temperature 300 --machine delta ../../data_generation/models/$m --start 10  --proteins bba chignolin homeodomain proteinb trpcage wwdomain
