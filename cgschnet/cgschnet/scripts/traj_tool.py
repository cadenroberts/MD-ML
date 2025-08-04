#!/usr/bin/env python3
import argparse
import mdtraj
import numpy as np
import os
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data.")
    parser.add_argument("input", nargs="+", help="Input files")
    parser.add_argument("-o", default="traj.pdb", help="Output path")
    parser.add_argument("-n", type=int, default=None, help="Approximate number of frames to include in the output the output")
    parser.add_argument("-r", type=str, default=None, help="Range of frames to select from in python slice syntax, e.g. 100:200")
    parser.add_argument("--offset", type=str, default=None, help="Offset between combined trajectories: x,y,z")
    # TODO: Add option not to center/superpose things

    args = parser.parse_args()

    target_num_frames = args.n
    save_filename = args.o

    if args.r:
        sel_slice = args.r.split(":")
        assert len(sel_slice) == 2
        sel_slice = slice(*[None if i == "" else int(i) for i in sel_slice])
    else:
        sel_slice = slice(None)

    for fn in args.input:
        assert os.path.exists(fn), f"{fn} does not exist."

    #TODO: Pick an offset based on the bounding sphere of the proteins
    offset_step = np.array([[0,0,4]])
    if args.offset:
        offset_step = np.array([[float(i) for i in args.offset.split(",")]])

    # if .npy file, load with pickle
    traj_list = []
    for fn in args.input:
        if fn.endswith(".npy"):
            struct = pickle.load(open(fn, "rb"))
            traj_list += struct['mdtraj_list']
            topology = struct['topology']
        else:
            traj_list += [mdtraj.load(fn)   [sel_slice]]

    print('trajlist', traj_list)

    if target_num_frames:
        traj_list = [i[::len(i)//target_num_frames] for i in traj_list]

    result = traj_list[0]
    result.center_coordinates()
    result.superpose(result)

    for cnt, traj in enumerate(traj_list[1:]):
        offset = offset_step*(cnt+1)
        traj.center_coordinates()
        traj.superpose(traj)
        traj.xyz += np.repeat(offset,traj.n_atoms,axis=0)
        result = result.stack(traj)
    result.save(save_filename)
    print("Trajectory saved:", save_filename)
