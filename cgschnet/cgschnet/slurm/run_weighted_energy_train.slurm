#!/usr/bin/env bash
#SBATCH --account=bbpa-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --mail-user=acbruce@ucsc.edu
#SBATCH --mail-type=ALL
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=4
#SBATCH --job-name=andy_openmm
#SBATCH --array=0-11

source /u/acbruce/.bashrc

micromamba activate cgschnet3


echo "TASK_ID", $SLURM_ARRAY_TASK_ID

# Start at 0.0, go up to 1.0 in steps of 0.1
energy_weight=$(bc <<< "scale=1; $SLURM_ARRAY_TASK_ID * 0.1")
force_weight=$(bc <<< "scale=1; 1.0 - $energy_weight")

echo "Running training with energy_weight=${energy_weight} and force_weight=${force_weight}"

python3 train.py \
	/work/hdd/bbpa/acbruce/training_data/chignolin_energy_2_components/preprocessed_chignolin_energy_2_tica_components \
	/work/hdd/bbpa/acbruce/models/energy_matching/results_${energy_weight}_${force_weight} \
	--gpus 0,1,2,3 \
	--atoms-per-call 140000 \
	--epoch 35 \
	--config=../configs/config_cutoff2.yaml \
	--wd=0 \
	--lr=0.001 \
	--exp-lr=0.85 \
	--batch=4 \
	--energy-weight="${energy_weight}" \
	--force-weight="${force_weight}"

# latest_folder=$(ls -td -- /media/DATA_18_TB_1/awaghili/data/*/ | head -n 1)
# ./gen_benchmark.py   --temperature 300   --machine bison   --proteins chignolin   -- "${latest_folder}" 
