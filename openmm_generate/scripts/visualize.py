#!/usr/bin/env python

import nglview as nv
import mdtraj as md
import os

def visualize_pdb(pdbid, basepath="/pscratch/sd/c/cawrober/benchmark_test"):
    pdb_file = os.path.join(basepath, pdbid, "simulation", "final_state.pdb")

    if not os.path.isfile(pdb_file):
        print(f"{pdbid}: FILE NOT FOUND")
        return None

    print(f"Loading: {pdb_file}")
    traj = md.load(pdb_file)

    # DNA bases
    dna_bases = ["DA", "DC", "DG", "DT", "A", "C", "G", "T"]
    dna_residues = [res for res in traj.topology.residues if res.name in dna_bases]

    # Protein residues
    protein_residues = [res for res in traj.topology.residues if res.name in [
        "ALA", "ARG", "ASN", "ASP", "CYS",
        "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO",
        "SER", "THR", "TRP", "TYR", "VAL"
    ]]

    if dna_residues:
        print(f"{pdbid}: Visualizing DNA only")
        atom_indices = [atom.index for res in dna_residues for atom in res.atoms]
    elif protein_residues:
        print(f"{pdbid}: No DNA → visualizing protein only")
        atom_indices = [atom.index for res in protein_residues for atom in res.atoms]
    else:
        print(f"{pdbid}: No DNA or protein found.")
        return None

    subset_traj = traj.atom_slice(atom_indices)

    view = nv.show_mdtraj(subset_traj)
    # view.clear_representations() uncomment for ribbons
    view.add_representation("cartoon")
    view.add_representation("licorice") # comment out for ribbons
    return view

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize DNA or protein only")
    parser.add_argument("pdbid", help="PDB ID folder name, e.g., 1D92")
    parser.add_argument("--basepath", default="/pscratch/sd/c/cawrober/benchmark_test", help="Base path")

    args = parser.parse_args()

    view = visualize_pdb(args.pdbid, args.basepath)
    if view:
        # This display works only in Jupyter, so in CLI it's a no-op.
        view.display()

