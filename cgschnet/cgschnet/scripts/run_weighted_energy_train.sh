#!/usr/bin/env bash

# Start at 0.0, go up to 1.0 in steps of 0.1
for i in {0..10}; do
  energy_weight=$(bc <<< "scale=1; $i * 0.1")
  force_weight=$(bc <<< "scale=1; 1.0 - $energy_weight")

  echo "Running training with energy_weight=${energy_weight} and force_weight=${force_weight}"

  python3 train.py \
      /media/DATA_18_TB_1/andy/preprocessed_data/preprocessed_chignolin_energy_2_tica_components \
      /media/DATA_18_TB_1/awaghili/data/results_${energy_weight}_${force_weight} \
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

  latest_folder=$(ls -td -- /media/DATA_18_TB_1/awaghili/data/*/ | head -n 1)
  ./gen_benchmark.py   --temperature 300   --machine bison   --proteins chignolin   -- "${latest_folder}" 
done
