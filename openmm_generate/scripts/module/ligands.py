from openmm.app import *
from openmm import *
from openmm.unit import *

from rdkit import Chem
from rdkit.Chem import AllChem

from openff.toolkit import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator

import json

from module import pdb_lookup

def replace_ligands(pdb_filename, modeller, smiles_templates=True, remove_ligands=False):
    """
    Insert or replace ligand molecules in modeller with corrected topology based on RCSB templates.

    Parameters:
    - pdb_filename (str): The path to the pdb file with ligands present
    - modeller (openmm.app.Modeller): The modeller structure to edit
    - smiles_templates (bool): If true return ligand templates as SMILEs strings,
                               if false return as openff.toolkit.Molecule objects
    - remove_ligands (bool): If true remove the ligands without generating templates

    Returns:
    - small_molecules (list()): A list of the ligand templates inserted
    """

    # We want to find any unbound small molecules
    pdb_mol = Chem.rdmolfiles.MolFromPDBFile(pdb_filename, removeHs=False, proximityBonding=True)

    standardResidues = {"ALA", "ARG", "ASN", "ASP", "ASX", "CYS", "GLU", "GLN", "GLX", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
    # Also ignore any explicit waters:
    standardResidues.add("HOH")

    # print("Extracting nonstandard residues...")
    fragments = dict()
    small_molecules_seen = dict()

    #TODO: Detect water by formula?
    #      * I'm not sure ifwater being named "HOH" is guranteed, so we should
    #        probably also check that the fragment's formula isn't H20
    for frag_idx, frag in enumerate(Chem.rdmolops.GetMolFrags(pdb_mol, asMols=True)):
        a = frag.GetAtomWithIdx(0)
        r_name = a.GetPDBResidueInfo().GetResidueName()

        # Skip over ions
        if frag.GetNumAtoms() == 1:
            # print(f"Skipped 1 atom fragment: {r_name}")
            continue
        if r_name in standardResidues:
            continue
        r_id = a.GetPDBResidueInfo().GetResidueNumber()
        r_chain = a.GetPDBResidueInfo().GetChainId()

        # We can only parameterize unbound molecules
        is_alone = True
        for a in frag.GetAtoms():
            if r_id != a.GetPDBResidueInfo().GetResidueNumber():
                is_alone = False
                break
        if is_alone:
            rcsb_smiles = pdb_lookup.get_rcsb_ligand_smiles(r_name)
            if rcsb_smiles is None:
                print(f"Could not find template for {r_name}")
                continue
            template = Chem.MolFromSmiles(rcsb_smiles)
            if smiles_templates:
                small_molecules_seen[r_name] = rcsb_smiles
            else:
                small_molecules_seen[r_name] = template
            frag = AllChem.AssignBondOrdersFromTemplate(template, frag)
            frag = Chem.AddHs(frag, addCoords=True)
            fragments[f"{r_chain}-{r_name}-{r_id}"] = frag

    if fragments:
        print(f"Found {len(fragments)} small molecules:", ", ".join(fragments.keys()))

    # This is unecessary but should 
    to_delete = []
    for residue in modeller.topology.residues():
        if residue.name not in standardResidues:
            query_key = f"{residue.chain.id}-{residue.name}-{residue.id}"
            if query_key in fragments:
                print("Removing", query_key)
                to_delete.append(residue)
    modeller.delete(to_delete) 

    if remove_ligands:
        return []

    # Using the unmodified templates because I don't know how the amber toolkit
    # parameterization interacts with conformations.
    if smiles_templates:
        small_molecules = list(small_molecules_seen.values())
    else:
        small_molecules = []
        for k, template in small_molecules_seen.items():
            print(f"Added {k} to small molecule templates.")
            #TODO: Should we add hydrogens? It doesn't seem to impact matching...
            template_mol = Molecule.from_rdkit(template, allow_undefined_stereo=True)
            small_molecules.append(template_mol)

    for k, frag in fragments.items():
        # The fragment still has the original residue metadata attached to it which
        # will confuse openmm's template system. Cycling to a mol block and back
        # strips this information out.
        frag_mol = Chem.MolToMolBlock(frag)
        frag_mol = Chem.MolFromMolBlock(frag_mol)
        frag_mol = Molecule.from_rdkit(frag_mol, allow_undefined_stereo=True)
        frag_top = frag_mol.to_topology()
        modeller.add(frag_top.to_openmm(), frag_top.get_positions().to_openmm())
        print(f"Added {k} to structure")

    return small_molecules

def add_ff_template_generator_from_json(forcefield, small_molecules_path, cache_path=None):
    """
    Add a GAFFTemplateGenerator to forcefield for the molecules listed in small_molecules.

    Parameters:
    - forcefield (openmm.app.ForceField): The forcefield object to add the generator to
    - small_molecules_smiles (list(str)): A list of SMILES strings to build templates for
    """
    with open(small_molecules_path, "r") as f:
        small_molecules_smiles = json.load(f)

    add_ff_template_generator_from_smiles(forcefield, small_molecules_smiles, cache_path)

def add_ff_template_generator_from_smiles(forcefield, small_molecules_smiles, cache_path=None):
    """
    Add a GAFFTemplateGenerator to forcefield for the molecules listed in small_molecules.

    Parameters:
    - forcefield (openmm.app.ForceField): The forcefield object to add the generator to
    - small_molecules_smiles (list(str)): A list of SMILES strings to build templates for
    """
    small_molecules = []
    for smiles in small_molecules_smiles:
        #TODO: Should we add hydrogens? It doesn't seem to impact matching...
        template = Chem.MolFromSmiles(smiles)
        template_mol = Molecule.from_rdkit(template, allow_undefined_stereo=True)
        small_molecules.append(template_mol)

    gaff = GAFFTemplateGenerator(molecules=small_molecules, cache=cache_path)
    forcefield.registerTemplateGenerator(gaff.generator)

    print(f"Added {len(small_molecules)} small molecule templates to forcefield")
