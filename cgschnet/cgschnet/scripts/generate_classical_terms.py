#!/usr/bin/env python
import mdtraj
import numpy as np
import os
import json
import glob
import tqdm
import argparse
import scipy.stats
import warnings
import multiprocessing

def calculate_classical_terms(src_path, output_dir, stride, mean_width, pad_value=-1000, term_names=["bond_d0", "angle_theta0", "dihedral_phi0"], read_semaphore=None, periodic=False):
    """
    stride: The stride used when generating the cg data from the high res data, e.g. every 50th frame
    mean_width: How many frames in either direction to include in the mean, so the slicing will be [i-mean_width:i+mean_width+1]
    pad_value: The fill value used if a mean can't be calculated (triggered when the mean width would include missing frames)
    """
    if read_semaphore:
        with read_semaphore:
            traj = mdtraj.load(src_path)
    else:
        traj = mdtraj.load(src_path)
    traj = traj.atom_slice(traj.top.select("name CA"))

    # FIXME: Would this be faster if we did `atom_idx = traj.top.select("name CA")` instead of slicing first?
    atom_idx = np.arange(traj.top.n_atoms)

    num_frames = len(traj)
    start_skip = int(np.ceil(mean_width/stride))
    end_skip = int(np.floor(mean_width/stride))

    if "bond_d0" in term_names:
        bonds = np.array([atom_idx[:-1], atom_idx[1:]]).T
        true_bonds = mdtraj.compute_distances(traj, bonds, periodic=periodic).T

        bond_means = []
        for bond_i in range(len(bonds)):
            bond_means_i = [np.mean(true_bonds[bond_i][i-mean_width:i+mean_width+1]) for i in range(start_skip*stride, num_frames-end_skip*stride, stride)]
            # Convert to nm -> angstroms
            bond_means_i = [i*10 for i in bond_means_i]
            # The first and last few strided frames won't have enough total frames to compute a mean, fill them with a pad value
            bond_means.append([pad_value]*start_skip + bond_means_i + [pad_value]*end_skip)
        
        # Ensure things get saved as a contiguous array for efficient reading
        bond_means = np.ascontiguousarray(np.array(bond_means).T)
        np.save(os.path.join(output_dir, "bond_d0.npy"), bond_means)

    if "angle_theta0" in term_names:
        angles = np.array([atom_idx[:-2], atom_idx[1:-1], atom_idx[2:]]).T
        true_angles = mdtraj.compute_angles(traj, angles, periodic=periodic).T
            
        angle_means = []
        for angle_i in range(len(angles)):
            angle_means_i = [np.mean(true_angles[angle_i][i-mean_width:i+mean_width+1]) for i in range(start_skip*stride, num_frames-end_skip*stride, stride)]
            # The first and last few strided frames won't have enough total frames to compute a mean, fill them with a pad value
            angle_means.append([pad_value]*start_skip + angle_means_i + [pad_value]*end_skip)

        angle_means = np.ascontiguousarray(np.array(angle_means).T)
        np.save(os.path.join(output_dir, "angle_theta0.npy"), angle_means)

    if "dihedral_phi0" in term_names:
        dihedrals = np.array([atom_idx[:-3], atom_idx[1:-2], atom_idx[2:-1],atom_idx[3:]]).T
        true_dihedrals = mdtraj.compute_dihedrals(traj, dihedrals, periodic=periodic).T

        num_frames = len(traj)
        start_skip = int(np.ceil(mean_width/stride))
        end_skip = int(np.floor(mean_width/stride))
            
        dihedral_means = []
        for dihedral_i in range(len(dihedrals)):
            # Pyright incorrectly infers the argument types for circmean - Daniel 2025.05.09
            dihedral_means_i = [scipy.stats.circmean(true_dihedrals[dihedral_i][i-mean_width:i+mean_width+1], high=np.pi, low=-np.pi) for i in range(start_skip*stride, num_frames-end_skip*stride, stride)] #pyright: ignore[reportArgumentType]
            # The first and last few strided frames won't have enough total frames to compute a mean, fill them with a pad value
            dihedral_means.append([pad_value]*start_skip + dihedral_means_i + [pad_value]*end_skip)

        diheral_means = np.ascontiguousarray(np.array(dihedral_means).T)
        np.save(os.path.join(output_dir, "dihedral_phi0.npy"), diheral_means)

    with open(os.path.join(output_dir, "term_gen.json"), "wt", encoding="utf-8") as f:
        info_dict = {
            "stride": stride,
            "mean_width": mean_width,
            "pad_value": pad_value,
            "terms": term_names,
        }
        json.dump(info_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate classical terms from a high-resolution dataset")
    parser.add_argument("input", type=str, help="Path to the input directory containing high-resolution data")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output directory to save the generated terms")
    parser.add_argument("--resume", action="store_true", help="Resume processing without overwritting already generated values")
    parser.add_argument("--stride", type=int, default=100, help="Stride used when generating the coarse-grained data from high-resolution data")
    parser.add_argument("--mean-width", type=int, default=50, help="Number of frames in either direction to include in the mean")
    parser.add_argument("--pad-value", type=float, default=-1000, help="Fill value used if a mean can't be calculated")
    parser.add_argument("--term-names", nargs="+", default=["bond_d0", "angle_theta0", "dihedral_phi0"], help="List of terms genearate (possible values: bond_d0, angle_theta0, dihedral_phi0)")
    parser.add_argument('--num-cores', type=int, default=8, help="Number of jobs to run in parallel")

    args = parser.parse_args()

    in_dir = args.input
    out_dir = args.output
    resume = args.resume
    num_cores = args.num_cores

    kwargs = {
        "stride": args.stride,
        "mean_width": args.mean_width,
        "pad_value": args.pad_value,
        "term_names": args.term_names,
        "periodic": False,
        "read_semaphore": multiprocessing.Semaphore(2),
    }

    warnings.filterwarnings(action="ignore", module="mdtraj", message="top= kwargs ignored since this file parser does not support it")

    assert os.path.isdir(in_dir), "Input directory does not exist"
    assert os.path.isdir(out_dir), "Output directory does not exist"

    def process_pdbid(pdbid):
        i_in_dir = os.path.join(in_dir, f"{pdbid}/result/output_{pdbid}.h5")
        i_out_dir = os.path.join(out_dir, f"{pdbid}/raw/")

        if not os.path.exists(i_out_dir):
            return f"Skipping {pdbid} (does not exist in output directory)"

        if resume and os.path.exists(os.path.join(i_out_dir, "term_gen.json")):
            return None

        calculate_classical_terms(i_in_dir, i_out_dir, **kwargs)
        return None

    pdbids = [os.path.basename(i) for i in glob.glob(os.path.join(in_dir, "*"))]
    with multiprocessing.Pool(num_cores) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(process_pdbid, pdbids), total=len(pdbids)):
            if result:
                tqdm.tqdm.write(result)