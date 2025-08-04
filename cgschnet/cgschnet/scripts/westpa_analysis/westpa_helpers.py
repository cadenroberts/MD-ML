import yaml
import json
import pickle
import os
import sys
import numpy as np
import mdtraj
import deeptime
import glob
import matplotlib.pyplot as plt
import tempfile
from tqdm import tqdm

def extract_simulation_config(cfg_path="west.cfg"):
    with open(cfg_path, "r") as file:
        yaml_content = file.read()
 
    config = yaml.load(yaml_content, Loader=yaml.UnsafeLoader)

    if not config or 'west' not in config:
        raise ValueError("YAML structure invalid or missing top-level 'west' key.")

    west_section = config['west']
    cg_prop = west_section.get('cg_prop', {})
    pcoord_calc = cg_prop.get('pcoord_calculator', {})

    extracted_data = {
        "model_path": cg_prop.get('model_path'),
        "cgschnet_path": cg_prop.get('cgschnet_path'),
        "topology_path": cg_prop.get('topology_path'),
        "components": pcoord_calc.get('components'),
        "tica_model_path": pcoord_calc.get('model_path')
    }

    return extracted_data

def extract_all_atom_simulation_config(cfg_path="west_openmm.cfg"):
    with open(cfg_path, "r") as file:
        yaml_content = file.read()
 
    config = yaml.load(yaml_content, Loader=yaml.UnsafeLoader)

    if not config or 'west' not in config:
        raise ValueError("YAML structure invalid or missing top-level 'west' key.")

    west_section = config['west']
    cg_prop = west_section.get('openmm', {})
    pcoord_calc = cg_prop.get('pcoord_calculator', {})

    extracted_data = {
        "model_path": cg_prop.get('model_path'),
        "cgschnet_path": cg_prop.get('cgschnet_path'),
        "topology_path": cg_prop.get('topology_path'),
        "components": pcoord_calc.get('components'),
        "tica_model_path": pcoord_calc.get('model_path')
    }

    return extracted_data

def convert_to_mdtraj_topology(cg_mol):
    with tempfile.TemporaryDirectory() as tmpdirname:
        topology_path = os.path.join(tmpdirname, "topology.pdb")
        cg_mol.write(topology_path)
        topology = mdtraj.load(topology_path).top
        return topology

def create_cg_topology_from_all_atom(simulation_config):
    cgschnet_path = simulation_config["cgschnet_path"] 
    if not cgschnet_path in sys.path:
        sys.path.append(cgschnet_path)
    import simulate  # pyright: ignore[reportMissingImports]

    checkpoint_path = simulation_config["model_path"] 

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
        prior_path, prior_params, simulation_config["topology_path"], use_box=False, verbose=False)

    return mol


def load_trajectories(coordinate_files, size_limit=None):
    coordinate_list = []
    label_list = []

    for cf in tqdm(coordinate_files):
        batch_label = os.path.basename(cf)
        batch_traj = []
        for subtraj in tqdm(glob.glob(cf)):
            if subtraj.endswith("npy"):
                coords = np.load(subtraj, allow_pickle=True)
                if type(coords) == dict: # One of Raz's benchmark archives
                    batch_label = os.path.join(*(cf.split(os.path.sep)[-2:]))
                    batch_traj.extend(coords["mdtraj_list"])
                else: # A preprocess.py output file
                    batch_label = os.path.join(*(cf.split(os.path.sep)[-3:]))
                    # Convert to NM to match mdtraj coordinates
                    coords = coords/10
                    psf_path = glob.glob(os.path.join(os.path.dirname(cf),"../processed/*_processed.psf"))[0]
                    traj = mdtraj.Trajectory(coords, topology=mdtraj.load_psf(psf_path))
                    batch_traj.append(traj)
            else: # Something mdtraj can open
                traj = mdtraj.load(subtraj)
                batch_traj.append(traj)
        if len(batch_traj) == 0:
            raise RuntimeError(f"{cf} did not match any files")
        
        batch_traj = mdtraj.join(batch_traj)

        # Select with a stride that brings the total number of frames down to the size_limit
        if size_limit and len(batch_traj) > size_limit:
            batch_traj = batch_traj[::(len(batch_traj)//size_limit)]

        label_list.append(batch_label)
        coordinate_list.append(batch_traj)

    assert len(coordinate_list) == len(label_list)
    return coordinate_list, label_list

def load_tica_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
        assert hasattr(model, "tica_model")
        return model

def calculate_component_values(model, coordinates, components):
    # Returns an object of type list(dict(array)) : [trajectory, component, component_values_for_frames]
    component_values = {k: [] for k in components}

    pairs = np.vstack(np.triu_indices(coordinates.n_atoms, k=1)).T
    distances = mdtraj.compute_distances(coordinates, pairs)
    tica_comps = model.tica_model.transform(distances)
    for k, v in component_values.items():
        v.extend(tica_comps[:, k])

    return component_values

def shorten_label(label, maxlen):
    if len(label) > maxlen:
        return label[:maxlen-3] + "..."
    return label
