#!/usr/bin/env python3
import os
import mdtraj
import json
import numpy as np
import pandas
import traceback

import torch
from tqdm import tqdm
from mdtraj.formats.hdf5 import HDF5TrajectoryFile

from simulate import load_model, External
import preprocess
from module import dataset
# from torchmd.forcefields.forcefield import ForceField
from module.torchmd.tagged_forcefield import TaggedYamlForcefield
from torchmd.forces import Forces
from torchmd.parameters import Parameters
from torchmd.systems import System

import numpy.typing
import numpy as np
import pickle
import os, sys

from report_generator.traj_loading import load_model_traj_pickle, ModelTraj
from report_generator.tica_plots import TicaModel, calc_atom_distance
from report_generator.reaction_coordinate import calc_reaction_coordinate, ReactionCoordKde
from report_generator.contact_maps import make_contact_map_plot, make_contact_map, ContactMap
from report_generator.bond_and_angle_analysis import plot_bond_length_angles, get_bond_angles
from report_generator.kullback_leibler_divergence import kl_div_calc, wasserstein
import scipy
import time
import mdtraj
from tabulate import tabulate

benchmark = "/media/DATA_18_TB_1/benchmark_sims/000206_results_0_1.0/benchmark.json"

def calc_energy(benchmark):
    with open(benchmark) as f:
        benchmark_data = json.loads(f.read())

        model_path = benchmark_data["model_path"]
        for protein_name, info in benchmark_data["proteins"].items():
            print('Processing protein: ', protein_name)
            protein_name: str
            model_trajs: list[ModelTraj] = load_model_traj_pickle(info["model_npy_path"])

            h5_paths = []
            if "native_paths" in info:
                for path_info in info["native_paths"]:
                    h5_path = path_info.get("h5_path")
                    if h5_path:
                        h5_paths.append(h5_path)

            res = compute_energies_for_input_paths(
                model_path=model_path,
                input_paths=h5_paths,
                csv_path="./save_energies.csv"
            )


            print(res)



def compute_potential_energy(mol, coords, box, prior_path, prior_params, calc):
    """
    Compute potential energy for each frame in 'coords'.

    Parameters
    ----------
    mol : Molecule-like object
        Molecule (PSF) from prior_builder.
    coords : np.ndarray
        Array of shape (n_frames, n_atoms, 3) in Angstroms.
    box : np.ndarray or None
        Array of shape (n_frames, 3, 3) in Angstroms or None if no box.
    prior_path : str
        Path to the 'priors.yaml'.
    prior_params : dict
        Dictionary loaded from 'prior_params.json'.
    calc : External or None
        If not None, it is the ML model wrapper to be added to the force terms.

    Returns
    -------
    np.ndarray
        Array of shape (n_frames,) containing the potential energy for each frame.
    """
    # If True, you can set your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    precision = torch.float
    replicas = 1

    natoms = mol.numAtoms

    # Convert coords to torch tensor
    coords = torch.tensor(coords, dtype=precision).contiguous().to(device)

    atom_vel = torch.zeros(replicas, natoms, 3)
    atom_pos = torch.zeros(natoms, 3, replicas)

    # Reshape or fill box appropriately
    if box is not None:
        # Flatten the 3x3 box, pick only the diagonal elements, then reshape
        linearized = box.reshape(-1, 9).take([0, 4, 8], axis=1)
        box_full = linearized.reshape(linearized.shape[0], 3, 1)
        if calc:
            calc.use_box = True
    else:
        box_full = torch.zeros(coords.shape[0], 3, 1)
        if calc:
            calc.use_box = False

    # Prepare the forcefield and parameters
    ff = TaggedYamlForcefield(mol, prior_path)
    parameters = Parameters(ff, mol, prior_params["forceterms"],
                            precision=precision, device=device)#pyright: ignore[reportArgumentType]
    forces = Forces(parameters,
                    terms=prior_params["forceterms"],
                    external=calc,
                    exclusions=prior_params["exclusions"])

    # Create the system object
    system = System(natoms, replicas, precision, device)
    system.set_positions(atom_pos)
    system.set_velocities(atom_vel)

    # Compute energies
    Epot_list = []
    n_frames = coords.shape[0] - 1
    for i in tqdm(range(n_frames), dynamic_ncols=True):
        system.set_box(box_full[i])
        Epot = forces.compute(coords[i : i+1], system.box, system.forces)
        Epot_list.append(Epot)

    return np.array(Epot_list)


def compute_energies_for_input_paths(model_path, input_paths, max_num_neighbors=None,
                                     prior_only=False, csv_path=None):
    """
    Load model + prior, then compute potential energies for each HDF5 input.
    Returns a dictionary (or optionally writes CSV) of mean, std, min, max energies.

    Parameters
    ----------
    model_path : str
        Path to the model or its directory (expects 'checkpoint.pth').
    input_paths : list of str
        List of HDF5 paths to process.
    max_num_neighbors : int or None
        Overrides 'max_num_neighbors' parameter of the model if not None.
    prior_only : bool
        If True, do not load the model, only the prior.
    csv_path : str or None
        If provided, save results to a CSV file at this path.

    Returns
    -------
    dict
        Dictionary containing arrays of 'path', 'mean', 'std', 'min', 'max' energies.
    """

    # If the user provided a directory, assume checkpoint is 'checkpoint.pth' inside it
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "checkpoint.pth")
    checkpoint_dir = os.path.dirname(model_path)

    prior_path = os.path.join(checkpoint_dir, "priors.yaml")
    if not os.path.exists(prior_path):
        raise FileNotFoundError(f"Could not find priors.yaml at {prior_path}")

    prior_params_path = os.path.join(checkpoint_dir, "prior_params.json")
    with open(prior_params_path, 'r') as file:
        prior_params = json.load(file)

    # Load the model unless prior_only
    model = None
    if not prior_only:
        model = load_model(model_path, device="cuda", max_num_neighbors=max_num_neighbors)

    results = {
        "path": [],
        "data": [],  
        "mean": [],
        "std": [],
        "min": [],
        "max": [],
    }

    for input_path in input_paths:
        try:
            print(f"\n=== Processing: {input_path} ===")
            pdb_path = os.path.join(os.path.dirname(input_path), "../simulation/final_state.pdb")

            # Construct the molecule from the prior
            prior_name = prior_params["prior_configuration_name"]
            prior_builder = preprocess.prior_types[prior_name]()
            print("Prior Config:", prior_name)
            print("Structure:   ", pdb_path)
            mol = prior_builder.write_psf(pdb_path, None)

            # Select relevant atoms from the PDB
            traj = mdtraj.load_frame(pdb_path, 0)
            atoms_idx = prior_builder.select_atoms(traj.top)
            embeddings = prior_builder.map_embeddings(atoms_idx, traj.top)

            # Load coordinates / box from the HDF5
            with HDF5TrajectoryFile(input_path) as f:
                # Example: read entire trajectory
                coords_nm = f.read_as_traj().xyz[:, atoms_idx, :]  # shape: (n_frames, n_sel_atoms, 3) #pyright: ignore[reportOptionalSubscript]
                coords = coords_nm * 10.0  # convert from nm to Angstroms
                traj = f.read_as_traj()

                # If you have box info:
                if traj.unitcell_lengths is not None:
                    boxes = traj.unitcell_vectors * 10 #pyright: ignore[reportOptionalOperand]
                else:
                    boxes = None

            # Build the 'calc' object if not using prior-only
            if prior_only:
                calc = None
            else:
                # If the representation model needs sequence or adjacency info, build it
                sequence = None
                rep_model = model.representation_model #pyright: ignore[reportOptionalMemberAccess]
                if hasattr(rep_model, "sequence_basis_radius") and rep_model.sequence_basis_radius != 0:
                    sequence = dataset.build_sequence_for_mol(mol)
                elif hasattr(rep_model, "adjacency_size") and rep_model.adjacency_size > 0:
                    sequence = dataset.build_adjacency_for_mol(mol, 3) #pyright: ignore[reportAttributeAccessIssue]

                calc = External(model, embeddings, device="cuda",
                                num_replicates=1, sequence=sequence)

            # Compute the potential energy for each frame
            epot_array = compute_potential_energy(mol, coords, boxes, prior_path, prior_params, calc)

            floats = epot_array.squeeze().tolist()  
            # Store summary
            results["path"].append(input_path)
            results["data"].append(floats)
            results["mean"].append(np.mean(epot_array))
            results["std"].append(np.std(epot_array))
            results["min"].append(np.min(epot_array))
            results["max"].append(np.max(epot_array))

            
            print("Energy stats [kcal/mol]:",
                  "mean =", results["mean"][-1],
                  "std =", results["std"][-1],
                  "min =", results["min"][-1],
                  "max =", results["max"][-1])

            if csv_path:
                df = pandas.DataFrame(results)
                df.to_csv(csv_path, index=False)


        except Exception as e:
            print("\nError processing", input_path)
            traceback.print_exc()
            results["path"].append(input_path)
            results["mean"].append(None)
            results["std"].append(None)
            results["min"].append(None)
            results["max"].append(None)

    return results


calc_energy(benchmark)
