# generally only use A40s on delta, as they cost 4X less credits. A40s are single-float precision, but that should not influence most of our workflows at all.

# interactive job, A40s only
intdelta:
	srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 \
--partition=gpuA40x4-interactive --account=bbpa-delta-gpu \
--gpus-per-node=4 --time=01:00:00 --pty /bin/bash
	cd cgschnet/scripts

# fast interactive job, A40s & A100
intdeltaf:
	srun  --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 \
--partition=gpuA40x4-interactive,gpuA100x4-interactive --account=bbpa-delta-gpu \
--gpus-per-node=4 --time=01:00:00 --pty /bin/bash

# fast interactive job, A40s & A100
intdeltaquad:
	srun  --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 \
--partition=gpuA100x8-interactive --account=bbpa-delta-gpu \
--gpus-per-node=8 --time=01:00:00 --pty /bin/bash


intnersc:
	salloc --nodes 1 --qos interactive --time 03:00:00 --constraint gpu --tasks-per-node=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 -c 32 --job-name=interactive --account=m4229

envcreate:
	micromamba env create -f /global/cfs/cdirs/m4229/raz/cgschnet/data_generation/jason.yml -n datagen

torchmdenv_create:
	#micromamba config append channels acellera
	#micromamba env create -n torchmdnet -f conda/raz_delta.yml # don't install this on delta, I got GLIBCXX_3.4.30 not found error
	#micromamba install -c conda-forge libstdcxx-ng
	micromamba env create -n torchmdnet -f conda/torchmdnet2_raz_delta.yaml # this worked out of the box on Dec 25th on Delta


	#then add the packages needed for datagen in the same env
	# update Dec 12: didn't work
	#conda env update -n torchmdnet -f conda/jason.yml

analyze:
	cd data_generation; python run.sh
	#python data_generation/analyze.py
	#python data_generation/analyze_seq.py

	# take all filtered PDBs from sequences2.txt and put them in a comma-separated list:
	cat sequences2.txt | awk '{printf "%s,", substr($1,1,4)} END {print ""}' | sed 's/,$//' > input_after_analyze_seq.txt



batch_generate:
	# check how many files you need to pre-process (e.g. 2000). then set batch-size to be ceil(2000/nr_array_jobs)
	sbatch batchgen_delta.slurm all_12368_cyrusc_081724
	#python3 batch_generate.py --input-dir /global/homes/r/razvanm/cgschnet/data_generation/saved_datasets/pdbs --batch-index 0 --batch-size 200 --pool-size 8 --gpus=0,1,2,3 --prepare --steps=1000000 --report-steps=1000 --data-dir /global/homes/r/razvanm/cgschnet/data_generation/saved_datasets/cyrusc_081924_small_multi_chain_mini/new_data --timeout 8


# with num-cores=32, it runs out of file descriptors on delta, with 16 it works
preprocess:
	# with the experimental prior by raz
	NUMEXPR_MAX_THREADS=64 python cgschnet/scripts/preprocess.py data_generation/saved_datasets/all_12368_cyrusc_081724/new_data  --output=data_generation/preprocessed_datasets/6k_flex --prior=CA_lj_angleXCX_dihedralX_flex --resume --num-cores=4

	#NUMEXPR_MAX_THREADS=64 python cgschnet/scripts/preprocess.py data_generation/saved_datasets/all_12368_cyrusc_081724/new_data  --output=data_generation/preprocessed_datasets/6k_flex --prior=CA_lj_angleXCX_dihedralX_flex --resume --num-cores=16 --jobid=0 --totalNrJobs=5

#pdb=2W9T
#pdb=2OKZ
pdb=6BA5
preprocessOne:
	python cgschnet/scripts/preprocess.py data_generation/saved_datasets/all_12368_cyrusc_081724/new_data  --output=data_generation/preprocessed_datasets/$(pdb)_flex --prior=CA_lj_angleXCX_dihedralX_flex --resume --num-cores=64 --pdbids=$(pdb)

preprocessStd:
	python cgschnet/scripts/preprocess.py data_generation/saved_datasets/all_12368_cyrusc_081724/new_data  --output=data_generation/preprocessed_datasets/$(pdb) --prior=CA_lj_angleXCX_dihedralX --resume --num-cores=64 --pdbids=$(pdb)

testDeltaForcesNN:
	# with the experimental prior by raz
	python cgschnet/scripts/test_deltaforces_nn.py data_generation/saved_datasets/all_12368_cyrusc_081724/new_data  --output=data_generation/preprocessed_datasets/dftest_$(pdb) --resume --num-cores=64 --pdbids=$(pdb)

bp=chignolin
simulatePriorOnlyNN:
	python cgschnet/scripts/simulate.py data_generation/models/all_12368_cyrusc_081724 /media/DATA_18_TB_1/andy/benchmark_set_5/$(bp)/starting_pos_100/processed/starting_pos_100_processed.pdb /media/DATA_18_TB_1/andy/benchmark_set_5/$(bp)/starting_pos_200/processed/starting_pos_200_processed.pdb --steps=1000 --save-steps=10 --output=cgschnet/scripts/sims/$(bp)_nn --prior-only --prior-nn=data_generation/preprocessed_datasets/6k_flex

simulatePriorOnlyNNdelta:
	python cgschnet/scripts/simulate.py data_generation/models/all_12368_cyrusc_081724 /work/hdd/bbpa/acbruce/md_data/generate/$(bp)/starting_pos_100/processed/starting_pos_100_processed.pdb /work/hdd/bbpa/acbruce/md_data/generate/$(bp)/starting_pos_200/processed/starting_pos_200_processed.pdb --steps=1000 --save-steps=10 --output=cgschnet/scripts/sims/$(bp)_nn --prior-only --prior-nn=data_generation/preprocessed_datasets/6k_flex

#envpy=/u/rmarinescu/micromamba/envs/torchmdnet3/bin/python
#d=all_12368_cyrusc_081724
#m=4k_v2 # same as the model below, but saved checkpoints every epoch
#m=all_12368_cyrusc_081724 was the original model trained on the 4k dataset
# note that --batch=4 means that a batch has 4 frames for each protein. For 4k proteins => batch size=4*4k=16k (4 almost maxes out the GPU memory). For the 4k protein dataset, we have 225 batches.
#python cgschnet/slurm/slurm_train_delta.py --email=ramarine@ucsc.edu --env-python=$(envpy)  --timeout=47:10:00 -- data_generation/preprocessed_datasets/$d  data_generation/models/$m --config=cgschnet/configs/config_cutoff2_seq6.yaml --wd=0 --lr=0.002 --exp-lr=0.90 --batch=4 --epochs=80 --checkpoint-save=1 --mini-epoch=25 --subsetpdbs=train_subset_ok.txt
train:
	sbatch cgschnet/slurm/train_delta.slurm # run this from the main cgschnet folder

benchmark_env:
	micromamba env create -n benchmark -f conda/andy_benchmark.yml
	micromamba activate benchmark
	
	git clone git@github.com:noegroup/aggforce.git
	pip install ./aggforce

benchmark:
	# use benchmark env
	cd cgschnet/scripts

	python3 ./gen_benchmark.py --temperature 300 --use-cache --machine bizon ../../data_generation/models/all_12368_cyrusc_081724  --proteins bba chignolin homeodomain trpcage wwdomain proteinb

benchmarkNNprior:
	# use benchmark env
	cd cgschnet/scripts

	python3 ./gen_benchmark.py ../../data_generation/models/all_12368_cyrusc_081724  --temperature 300  --machine delta --use-cache   --proteins chignolin --prior-only --prior-nn ../../data_generation/preprocessed_datasets/6k_flex

report:
	# use benchmark env
	cd cgschnet/scripts
	python gen_report.py sims/000037_all_12368_cyrusc_081724/benchmark.json --also-plot-locally
	

preprocessTest:
	python cgschnet/scripts/preprocess.py data_generation/saved_datasets/all_12368_cyrusc_081724/test_data  --output=data_generation/preprocessed_datasets/6k_test6 --prior=CA_lj_angleXCX_dihedralX --resume --num-cores=12