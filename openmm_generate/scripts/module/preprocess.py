import openmm as mm
import openmm.app as app
from openmm import unit
import pdbfixer
import requests
import json
import os
import shutil
from module import ligands
from module import function


def prepare_protein(pdbid=str, remove_ligands=False, implicit_solvent=False):
    """
    Preprocesses a protein by downloading the PDB file, fixing missing residues and atoms,
    adding missing hydrogens, adding solvent, and writing the processed PDB file.

    Args:
        pdbid (str): The PDB ID of the protein.

    Returns:
        None
    """

    local_input_pdb_path = None
    if pdbid.endswith(".pdb"):
        local_input_pdb_path = pdbid
        pdbid = os.path.splitext(os.path.basename(local_input_pdb_path))[0]

    print(f"Preprocess of {pdbid}")
    # create folder
    function.create_folder(function.get_data_path(pdbid))
    pdb_path = function.get_data_path(f"{pdbid}/raw/{pdbid}.pdb")

    if local_input_pdb_path:
        print("Using local input:", local_input_pdb_path)
        shutil.copyfile(local_input_pdb_path, pdb_path)
    else:
        pdb_url = f"https://files.rcsb.org/download/{pdbid}.pdb"

        # download pdb file
        if not os.path.exists(pdb_path):
            r = requests.get(pdb_url)
            r.raise_for_status()
            with open(pdb_path, "wb") as f:
                f.write(r.content)
        else:
            print(f"{pdbid}.pdb already downloaded")

    fixer = pdbfixer.PDBFixer(pdb_path)

    # find missing residues and atoms
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    print(f"Missing residues: {fixer.missingResidues}")
    print(f"Missing terminals: {fixer.missingTerminals}")
    print(f"Missing atoms: {fixer.missingAtoms}")

    # remove missing residues at the terminal
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain = chains[key[0]]
        # terminal residues
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]

    # check if the terminal residues are removed
    for key in list(keys):
        chain = chains[key[0]]
        assert key[1] != 0 or key[1] != len(list(chain.residues())), "Terminal residues are not removed."

    # remove ligand molecules if requested
    if remove_ligands:
        fixer.removeHeterogens(True)

    # find nonstandard residues
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    # add missing atoms, residues, and terminals
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # add missing hydrogens
    ph = 7.0
    fixer.addMissingHydrogens(ph)

    # make modeller
    modeller = app.Modeller(fixer.topology, fixer.positions)

    # add ligands back to the prepaired protein
    if remove_ligands:
        small_molecules = None
    else:
        small_molecules = ligands.replace_ligands(pdb_path, modeller, remove_ligands=False)

    print("\nAfter the process")
    print(f"Missing residues: {fixer.missingResidues}")
    print(f"Missing terminals: {fixer.missingTerminals}")
    print(f"Missing atoms: {fixer.missingAtoms}")

    # set the forcefield
    if implicit_solvent:
        forcefield_configs = ["amber14-all.xml", "implicit/gbn2.xml"]
    else:
        forcefield_configs = ["amber14-all.xml", "amber14/tip3pfb.xml"]
    json.dump(forcefield_configs, open(function.get_data_path(f'{pdbid}/processed/forcefield.json'), 'w', encoding='utf-8'))

    forcefield = app.ForceField(*forcefield_configs)

    if small_molecules:
        json.dump(small_molecules, open(function.get_data_path(f'{pdbid}/processed/{pdbid}_processed_ligands_smiles.json'), 'w'))
        template_cache_path = function.get_data_path(f'{pdbid}/processed/{pdbid}_processed_ligands_cache.json')
        ligands.add_ff_template_generator_from_smiles(forcefield, small_molecules, cache_path=template_cache_path)

    if implicit_solvent:
        modeller.deleteWater()

    # Small molecules we've added templates for will be named "UNK"
    unmatched_residues = [r for r in forcefield.getUnmatchedResidues(modeller.topology) if r.name != "UNK"]
    if unmatched_residues:
        raise RuntimeError("Structure still contains unmatched residues after fixup: " + str(unmatched_residues))

    # Add the water molecules if we're using explicit solvent
    if not implicit_solvent:
        modeller.addSolvent(forcefield, padding=1.0 * unit.nanometers, ionicStrength=0.15 * unit.molar)

    # write the processed pdb file & ligand templates
    top = modeller.getTopology()
    pos = modeller.getPositions()
    app.PDBFile.writeFile(top, pos, open(function.get_data_path(f'{pdbid}/processed/{pdbid}_processed.pdb'), 'w'))

    # Validate that the PDB file contains the correct structure
    pdb = app.PDBFile(function.get_data_path(f'{pdbid}/processed/{pdbid}_processed.pdb'))
    assert pdb.topology.getNumResidues() == top.getNumResidues()
    assert pdb.topology.getNumAtoms() == top.getNumAtoms()
    assert pdb.topology.getNumBonds() == top.getNumBonds()

    top_bonds = [len([*i.bonds()]) for i in top.residues() if i.name == 'UNK']
    pdb_bonds = [len([*i.bonds()]) for i in pdb.topology.residues() if i.name == 'UNK']
    assert pdb_bonds == top_bonds