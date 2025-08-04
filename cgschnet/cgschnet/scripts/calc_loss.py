#!/usr/bin/env python3
import os
import mdtraj
import json
import numpy as np
import pandas
import traceback

import torch
# from torchmd.forcefields.forcefield import ForceField
from module.torchmd.tagged_forcefield import TaggedYamlForcefield
from torchmd.forces import Forces
from torchmd.parameters import Parameters
from torchmd.systems import System
from tqdm import tqdm

import torch.nn as nn

from mdtraj.formats.hdf5 import HDF5TrajectoryFile

from simulate import load_model, External
import preprocess
from module import dataset

def compute_loss(mol, coords, box, all_forces, prior_path, prior_params, calc):
    device = torch.device("cuda")
    precision = torch.float
    replicas = 1

    natoms = mol.numAtoms

    #FIXME: Unsure why we need .contiguous() here...
    coords = torch.tensor(coords, dtype=precision).contiguous().to(device)
    all_forces = torch.tensor(all_forces, dtype=precision).to(device)

    atom_vel = torch.zeros(replicas, natoms, 3)
    atom_pos = torch.zeros(natoms, 3, replicas)
    if box is not None:
        # Reshape box to be rectangle, then format to be given to set_box
        linearized = box.reshape(-1,9).take([0,4,8],axis=1)
        box_full = linearized.reshape(linearized.shape[0], 3, 1)

        if calc:
            calc.use_box = True
    else:
        box_full = torch.zeros(coords.shape[0], 3, 1)

        if calc:
            calc.use_box = False

    ff = TaggedYamlForcefield(mol, prior_path)
    parameters = Parameters(ff, mol, prior_params["forceterms"], precision=precision, device=device) # pyright: ignore[reportArgumentType]
    forces = Forces(parameters, terms=prior_params["forceterms"], external=calc, exclusions=prior_params["exclusions"])

    system = System(natoms, replicas, precision, device)
    system.set_positions(atom_pos)
    system.set_velocities(atom_vel)

    criterion = nn.MSELoss()

    val_loss_list = []
    for i in tqdm(range(0, coords.shape[0]), dynamic_ncols=True):
        system.set_box(box_full[i])
        Epot = forces.compute(coords[i:i+1], system.box, system.forces)
        out = system.forces[0]
        val_loss_list.append(criterion(out, all_forces[i]).item())

    return np.array(val_loss_list)


def main():
    import argparse
    arg_parser = argparse.ArgumentParser(description="Cacluate the model loss on an unprocessed all atom trajectory")
    arg_parser.add_argument("-m", "--model", required=True, help="The model or checkpoint to use")
    arg_parser.add_argument("input_paths", nargs="+", help="The all atom h5 files to process")
    arg_parser.add_argument("--max-num-neighbors", default=None, type=int, help="Override the 'max_num_neighbors' parameter of the model")
    arg_parser.add_argument("--prior-only", default=False, action='store_true', help="Disable the model and use only the prior forcefield")
    arg_parser.add_argument("--csv", default=None, help="Save results to this path in a csv file")

    args = arg_parser.parse_args()
    print(args)

    checkpoint_path = args.model
    device = "cuda"

    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "checkpoint.pth")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    prior_path = os.path.join(checkpoint_dir, "priors.yaml")
    assert os.path.exists(prior_path)
    prior_params_path = os.path.join(checkpoint_dir, "prior_params.json")

    # Load forcefield terms
    with open(f"{prior_params_path}", 'r') as file:
        prior_params = json.load(file)

    # Load the model
    model = load_model(checkpoint_path, device, max_num_neighbors=args.max_num_neighbors)

    results = {
        "path": [],
        "mean": [],
        "std": [],
        "min": [],
        "max": [],
    }

    for input_path in args.input_paths:
        try:
            print()
            pdb_path = os.path.join(os.path.dirname(input_path), "../simulation/final_state.pdb")

            # Generate a mol based on the prior
            prior_name = prior_params["prior_configuration_name"]
            prior_builder = preprocess.prior_types[prior_name]()
            print("Prior Config:", prior_name)
            print("Structure:   ", pdb_path)
            mol = prior_builder.write_psf(pdb_path, None)
            traj = mdtraj.load_frame(pdb_path, 0)
            atoms_idx = prior_builder.select_atoms(traj.top)
            embeddings = prior_builder.map_embeddings(atoms_idx, traj.top)

            # Load and course grain the trajectory
            #TODO: Implement slicing down to a subset of frames
            frame_slice = slice(None, None)
            with HDF5TrajectoryFile(input_path) as f:
                forces = f.root["forces"][frame_slice, atoms_idx, :]
                # Convert from kilojoules/mole/nanometer to kilocalories/mole/angstrom
                forces = forces*0.02390057361376673
                traj = f.read_as_traj()
                # Convert distances from nm to Ang
                coords = traj.xyz[frame_slice, atoms_idx, :] * 10 #pyright: ignore[reportOptionalSubscript]
                if traj.unitcell_lengths is not None:
                    boxes = traj.unitcell_vectors * 10 #pyright: ignore[reportOptionalOperand]
                else:
                    boxes = None

            if args.prior_only:
                print("Prior only mode...")
                calc = None
            else:
                sequence = None
                if hasattr(model.representation_model, "sequence_basis_radius") and \
                model.representation_model.sequence_basis_radius != 0:
                    print("Generating sequence info...")
                    sequence = dataset.build_sequence_for_mol(mol)
                elif hasattr(model.representation_model, "adjacency_size") and \
                model.representation_model.adjacency_size > 0:
                    print("Generating adjacency info...")
                    sequence = dataset.build_adjacency_for_mol(mol, 3)#pyright: ignore[reportAttributeAccessIssue]
                calc = External(model, embeddings, device, num_replicates=1, sequence=sequence)

            val_loss_list = compute_loss(mol, coords, boxes, forces, prior_path, prior_params, calc)
            print()
            results["path"].append(input_path)
            results["mean"].append(np.mean(val_loss_list))
            results["std"].append(np.std(val_loss_list))
            results["min"].append(np.min(val_loss_list))
            results["max"].append(np.max(val_loss_list))
            print("Loss:", results["mean"][-1], "std:", results["std"][-1], "min:", results["min"][-1], "max:", results["max"][-1])
        except Exception as e:
            print()
            traceback.print_tb(e.__traceback__)
            print(e)
            results["path"].append(input_path)
            results["mean"].append(None)
            results["std"].append(None)
            results["min"].append(None)
            results["max"].append(None)

    if args.csv:
        df = pandas.DataFrame(results)
        df.to_csv(args.csv, index=False)

if __name__ == "__main__":
    main()
