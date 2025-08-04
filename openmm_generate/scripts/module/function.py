from openmm.app.pdbfile import PDBFile
import numpy as np
import os

_data_dir_path = "../data/"

def get_non_water_atom_indexes(topology):
    """
    Get the atom indices for all non-water residues in the topology.

    Returns:
    - indices (numpy.ndarray): An array of atom indices from protein.
    """
    return np.array([a.index for a in topology.atoms() if a.residue.name != 'HOH'])

def create_folder(folder_path):
    """
    Create a folder if it does not exist, or do nothing if it already exists.
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            os.makedirs(f"{folder_path}/raw")
            os.makedirs(f"{folder_path}/interim")
            os.makedirs(f"{folder_path}/processed")
            os.makedirs(f"{folder_path}/result")
            os.makedirs(f"{folder_path}/simulation")
            
            print(f"Folder created: {folder_path}")
        except OSError as e:
            print(f"Error: Unable to create folder {folder_path}. {e}")
    else:
        print(f"Folder already exists: {folder_path}")

def set_data_dir(path):
    """
    Set the location of the data directory ("../data/" by default).

    Args:
        path (str): The path to the data directory.
    """
    global _data_dir_path
    _data_dir_path = path

def get_data_path(path=None):
    """
    Get the full path of "path" inside the current data directory.

    Args:
        path (str): The subdirectory append to the data directory path,
                    or None to get the current data direcotry.
    Returns:
        str: The combined path.
    """
    if path is None:
        path = ""
    return os.path.join(_data_dir_path, path)

def get_atomSubset(pdb_path=str):
    """
    Get the subset of atom indices for protein residues in a PDB file.
    
    Args:
        pdb_path (str): The path to the PDB file.
        
    Returns:
        list: A list of atom indices corresponding to protein residues.
    """
    
    proteinResidues = ['ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR', 'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL']
    
    pdb = PDBFile(pdb_path)

    atomSubset = []
    topology = pdb.getTopology()
    for atom in topology.atoms():
        if atom.residue.name in proteinResidues:
            atomSubset.append(atom.index)
    
    return atomSubset