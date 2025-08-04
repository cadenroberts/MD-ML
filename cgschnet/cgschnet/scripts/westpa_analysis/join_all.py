#!/usr/bin/env python3
import sys
import os
import subprocess
import tempfile
import glob
import numpy as np
from tqdm import tqdm
import h5py
import mdtraj
import json
from collections import defaultdict

def append_seg(seg_dir, traj):
    try:
        bstate_path = os.path.join(seg_dir, 'bstate.pdb')
        dcd_path = os.path.join(seg_dir, 'seg.dcd')
        return traj + mdtraj.load_dcd(dcd_path, top=bstate_path)
    except Exception as e:
        print("Error appending ", seg_dir)
        print(e)
        return traj

def convert_to_mdtraj_topology(cg_mol):
    with tempfile.TemporaryDirectory() as tmpdirname:
        topology_path = os.path.join(tmpdirname, "topology.pdb")
        cg_mol.write(topology_path)
        topology = mdtraj.load(topology_path).top
        return topology

def get_bstate_traj(sim_dir):
    checkpoint_path = "/media/DATA_18_TB_1/daniel_s/cgschnet/harmonic_net_2025.04.06/model_high_density_benchmark_CA_only_2025.04.03_mix1_s100_CA_lj_bondNull_angleNull_dihedralNull_cutoff2_seq6_harBAD_termBAD100__wd0_plateaulr5en4_0.1_0_1en3_1en7_bs4_chunk120"
    topology_file = "/media/DATA_18_TB_1/andy/benchmark_set_5/trpcage/starting_pos_0/processed/starting_pos_0_processed.pdb"  
    cgschnet_path = "/home/md-ml/awaghili/cgschnet/cgschnet/scripts"
    if not cgschnet_path in sys.path:
        sys.path.append(cgschnet_path)
    import simulate  # pyright: ignore[reportMissingImports]


    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "checkpoint-best.pth")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    assert os.path.exists(checkpoint_path)

    prior_path = os.path.join(checkpoint_dir, "priors.yaml")
    assert os.path.exists(prior_path)
    prior_params_path = os.path.join(checkpoint_dir, "prior_params.json")

    with open(f"{prior_params_path}", 'r') as file:
        prior_params = json.load(file)

    mol, embeddings = simulate.load_molecule(
        prior_path, prior_params, topology_file, use_box=False, verbose=False)

    return convert_to_mdtraj_topology(mol) 

def get_trace_directories(rc_file, sim_dir, traj_id):
    dirstash = os.getcwd()

    traj_dirs = []

    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        # print(os.getcwd())
        # print(os.environ["WEST_SIM_ROOT"])
        westh5path = os.path.join(sim_dir, "west.h5")
        subprocess.run(["w_trace", "-r", rc_file, "-W", westh5path, traj_id])

        final_iter_id, final_seg_id = [int(i) for i in traj_id.split(":")]
        with open(f"traj_{final_iter_id}_{final_seg_id}_trace.txt", "rt", encoding="utf-8") as trace_file:
            lines = [l for l in trace_file.read().split("\n") if not l.startswith("#") and l]
            # lines = trace_file.read().split("\n")[6:-1]
            for line in lines:
                # print(line.split()[:2])
                iter_id, seg_id = map(int, line.split()[:2])
                traj_dirs.append(os.path.join(sim_dir, "traj_segs", f"{iter_id:06d}", f"{seg_id:06d}"))
                print(traj_dirs[-1])

        os.chdir(dirstash)
    
    return traj_dirs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_dir", type=str, help="WEST_SIM_ROOT")
    # parser.add_argument("traj_id", type=str, help="iter:seg")
    parser.add_argument("-r", "--rc-file", type=str, default="west.cfg", help="WESTPA RC file location")
    parser.add_argument("-o", type=str, help="Output filename")

    args = parser.parse_args()
    print(args)

    args.sim_dir

    # conf_path = os.path.realpath(args.rc_file)
    # print(conf_path)
    # This must be set to correctly parse the RC file
    # os.environ["WEST_SIM_ROOT"] = args.sim_dir

    # traj_dirs = get_trace_directories(args.rc_file, args.sim_dir, args.traj_id)
    # # FIXME: The first directory isn't real (0 is the bstate), do we want to include it some how?
    # #        -> Currently the first bstate is used in merge_segments, but we might want to allow more that one bstate
    # traj_dirs = traj_dirs[1:]

    if args.o:
        merged_name = args.o
    else:
        name_prefix = os.path.basename(os.path.normpath(args.sim_dir))
        merged_name = "combined_trajs/trpcage"

    sim_dir = args.sim_dir
    topology = get_bstate_traj(sim_dir)
    traj_dirs = sorted(glob.glob(f"{sim_dir}/traj_segs/*/*"))

    # We don't include the base trajectory frame because we have no forces for it
    forces = []
    positions = []
    times = []
    
    i = 0

    iteration_groups = defaultdict(list)
    for td in traj_dirs:
        parts = td.split(os.sep)
        iteration = parts[-2]  # assuming path: traj_segs/iteration#/segment#
        iteration_groups[iteration].append(td)

    for i, iteration in tqdm(enumerate(sorted(iteration_groups.keys()))):
        for td in iteration_groups[iteration]:
            try:
                seg_npz_data = np.load(os.path.join(td, "seg.npz"))
            except Exception as e:
                print("Error appending ", td)
                print(e)
                continue
            positions.append(seg_npz_data["pos"] * 0.1)
            times.append(seg_npz_data["time"])

        if positions:
            pos = np.concatenate(positions)
            t = np.concatenate(times)
            traj = mdtraj.Trajectory(xyz=pos, topology=topology)
            traj.save_hdf5(f"{merged_name}_{i}.h5")

    pos = np.concatenate(positions)
    traj = mdtraj.Trajectory(xyz=pos, topology=topology)
    traj.save_hdf5(merged_name + "_all.h5")
    print("Saved to:", merged_name)




