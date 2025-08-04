import pickle
import mdtraj
cg_poses = pickle.load(open("/media/DATA_18_TB_1/benchmark_sims/000081_result-2025.01.15-10.38.13/chignolin_model_replicas.npy", "rb"))

for i, traj in enumerate(cg_poses["mdtraj_list"]):
    traj.save(f"stuff_{i}.h5")
