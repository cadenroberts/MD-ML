import numpy as np
import logging
import moleculekit.readers
import moleculekit.molecule
from moleculekit.molecule import Molecule
from .torchmd_cg_mappings import CA_MAP, CACB_MAP
import mdtraj

# adapted from https://github.com/torchmd/torchmd-cg/blob/master/torchmd_cg/utils/psfwriter.py


def pdb2psf_CA(pdb_name_in, psf_name_out, bonds=True, angles=True, dihedrals=False, tag_beta_turns=True):

        moleculekit_readers_level = moleculekit.readers.logger.level
        moleculekit_molecule_level = moleculekit.molecule.logger.level
        # Setting the readers to "ERROR" is sketchy but the multiprocess output is useless otherwise...
        #TODO: Redirect these logs somewhere sane instead of silencing them
        moleculekit.readers.logger.setLevel(logging.ERROR)
        moleculekit.molecule.logger.setLevel(logging.WARNING)

        # If the input is a trajectory load only the 1st frame
        mol = Molecule(pdb_name_in, frames=[0])
        # Get carbon alpha atoms from protein
        mol.filter("name CA and element C")

        moleculekit.readers.logger.setLevel(moleculekit_readers_level)
        moleculekit.molecule.logger.setLevel(moleculekit_molecule_level)

        # Clean up the PSF file to ensure mdtraj will be happy with it:
        # Make the atom ids sequential
        mol.serial = np.arange(len(mol.serial))+1
        # Remove insertions (if the pdb resid is something like '17X' the 'X' is the insertion)
        mol.insertion = np.repeat(np.array(str(''), dtype=object), len(mol.insertion)) #pyright: ignore[reportAttributeAccessIssue]

        n = mol.numAtoms

        atom_types = []
        for i in range(n):
            atom_types.append(CA_MAP[(mol.resname[i], mol.name[i])]) #pyright: ignore[reportAttributeAccessIssue]

        # We use mdtraj for this step because its topology functions are easier to work with
        traj = mdtraj.load_frame(pdb_name_in, 0)
        traj_ca = traj.atom_slice(traj.top.select("name CA and element C"))

        assert traj_ca.top.n_atoms == mol.numAtoms

        all_bonds = []
        beta_turn_atoms = []
        for c_id in range(traj_ca.top.n_chains):
            chain_ca = traj_ca.top.select(f"name CA and chainid {c_id}")
            if len(chain_ca) > 0: # Otherwise it's probably a ligands / ions
                assert len(chain_ca) == traj.top.chain(c_id).n_residues
                # Make a list of each sequential pair of CAs in the chain
                chain_bonds = np.array([chain_ca[:-1], chain_ca[1:]]).T

                # Overly long bonds indicate a gap in the chain
                bond_lengths = mdtraj.compute_distances(traj_ca, chain_bonds)[0]
                if tag_beta_turns:
                    # Use a somewhat arbitrary cutoff of 0.35 nm to detect beta turns
                    for bond in chain_bonds[bond_lengths < 0.35]:
                        # And require that it involves proline...
                        if atom_types[bond[0]] == "CAP" or atom_types[bond[1]] == "CAP":
                            beta_turn_atoms.extend(bond)
                chain_bonds = chain_bonds[bond_lengths < 1]
                all_bonds.extend(chain_bonds)

        if tag_beta_turns:
            assert len(beta_turn_atoms) == len(set(beta_turn_atoms)), "Atoms can't be tagged twice"
            for i in beta_turn_atoms:
                atom_types[i] = atom_types[i] + "*"

        if bonds:
            bonds_to_write = np.array(all_bonds, dtype=np.int32)
        else:
            bonds_to_write = np.empty([0, 2], dtype=np.int32)

        # The angles and dihedrals logic relies on the fact that bonds within CA chain will be sequential
        if angles:
            angles_to_write = []
            # Each CA-CA-CA angle corresponds to two sequential bonds
            for i in range(len(all_bonds)-1):
                if all_bonds[i][1] == all_bonds[i+1][0]:
                    angles_to_write.append([all_bonds[i][0], all_bonds[i][1], all_bonds[i+1][1]])
            angles_to_write = np.array(angles_to_write, dtype=np.int32)
        else:
            angles_to_write = np.empty([0, 3], dtype=np.int32)

        if dihedrals:
            dihedrals_to_write = []
            # Make a dihedral for each bonded sequence of CA-CA-CA-CA
            for i in range(len(all_bonds)-2):
                if (all_bonds[i][1]   == all_bonds[i+1][0]) and (all_bonds[i+1][1] == all_bonds[i+2][0]):
                    dihedrals_to_write.append([
                        all_bonds[i][0], all_bonds[i][1],
                        all_bonds[i+2][0], all_bonds[i+2][1],
                    ])
            dihedrals_to_write = np.array(dihedrals_to_write, dtype=np.int32)
        else:
            dihedrals_to_write = np.empty([0, 4], dtype=np.int32)

        mol.atomtype = np.array(atom_types) #pyright: ignore[reportAttributeAccessIssue]
        # The mass of carbon used here is the from OpenMM/AMBER-14 value
        mol.masses = np.repeat(np.array(12.01, dtype=np.float32), len(mol.atomtype)) #pyright: ignore[reportAttributeAccessIssue]
        mol.box = np.zeros((3, 0), dtype=np.float32)
        mol.bonds = bonds_to_write
        mol.angles = angles_to_write #pyright: ignore[reportAttributeAccessIssue]
        mol.dihedrals = dihedrals_to_write #pyright: ignore[reportAttributeAccessIssue]
        if psf_name_out is not None:
            mol.write(psf_name_out)
        return mol

def pdb2psf_CACB(pdb_name_in, psf_name_out, bonds=True, angles=True, dihedrals=False):
    mol = Molecule(pdb_name_in, frames=[0])

    moleculekit_level = moleculekit.molecule.logger.level
    moleculekit.molecule.logger.setLevel(30)
    mol.filter("name CA CB and element C")
    moleculekit.molecule.logger.setLevel(moleculekit_level)

    # Clean up the PSF file to ensure mdtraj will be happy with it:
    # Make the atom ids sequential
    mol.serial = np.arange(len(mol.serial))+1
    # Remove insertions (if the pdb resid is something like '17X' the 'X' is the insertion)
    mol.insertion = np.repeat(np.array(str(''), dtype=object), len(mol.insertion)) #pyright: ignore[reportAttributeAccessIssue]

    n = mol.numAtoms

    atom_types = []
    for i in range(n):
        atom_types.append(CACB_MAP[(mol.resname[i], mol.name[i])]) #pyright: ignore[reportAttributeAccessIssue]

    # We use mdtraj for this step because its topology functions are easier to work with
    traj = mdtraj.load_frame(pdb_name_in, 0)
    traj_sel = traj.atom_slice(traj.top.select("name CA CB and element C"))

    assert traj_sel.top.n_atoms == mol.numAtoms

    ca_bonds = []
    cb_bonds = []
    ca_angles = []
    cb_angles = []
    dihedral_angles = []

    for chain in traj_sel.top.chains:
        CA_idx = []
        CB_idx = []
        for residue in chain.residues:
            CA_idx.append(next(residue.atoms_by_name("CA")).index)
            cb_atom = next(residue.atoms_by_name("CB"), None)
            if cb_atom is not None:
                CB_idx.append(cb_atom.index)
            else:
                CB_idx.append(None)

        # Make CA-CA bonds
        chain_bonds = np.array([CA_idx[:-1], CA_idx[1:]]).T
        # Overly long CA-CA bonds indicate a gap in the chain
        bond_lengths = mdtraj.compute_distances(traj_sel, chain_bonds)[0]
        chain_bonds = chain_bonds[bond_lengths < 1]
        ca_bonds.extend(chain_bonds)

        # Make CA-CB bonds
        for i in range(len(CA_idx)):
            if CB_idx[i] is not None:
                cb_bonds.append([CA_idx[i], CB_idx[i]])

        # Each are two CA-CA-CB angles per residue pair (3,1,2) and (1,3,4) below:
        # 2  4  6
        # |  |  |
        # 1--3--5
        # Additionally there is the CA-CA-CA angle: (1,3,5)
        # And the dihedral: (2,1,3,4)

        # Each CA-CA-CA angle corresponds to two sequential bonds
        for i in range(len(chain_bonds)-1):
            if chain_bonds[i][1] == chain_bonds[i+1][0]:
                ca_angles.append([chain_bonds[i][0], chain_bonds[i][1], chain_bonds[i+1][1]])

        # Make the CA-CA-CB angles & CB-CA-CA-CB dihedrals
        for i in range(len(CA_idx) - 1):
            # Don't make angles across gaps in the chain
            if bond_lengths[i] >= 1:
                continue
            # The left angle
            if CB_idx[i] is not None:
                cb_angles.append([CA_idx[i+1], CA_idx[i], CB_idx[i]])
            # The right angle
            if CB_idx[i+1] is not None:
                cb_angles.append([CA_idx[i], CA_idx[i+1], CB_idx[i+1]])
            # The dihedral
            if (CB_idx[i] is not None) and (CB_idx[i+1] is not None):
                dihedral_angles.append([CB_idx[i], CA_idx[i], CA_idx[i+1], CB_idx[i+1]])

    if bonds:
        bonds_to_write = np.concatenate((np.array(ca_bonds), np.array(cb_bonds)), dtype=np.int32)
    else:
        bonds_to_write = np.empty([0, 2], dtype=np.int32)

    if angles:
        angles_to_write = np.concatenate((np.array(ca_angles), np.array(cb_angles)), dtype=np.int32)
    else:
        angles_to_write = np.empty([0, 3], dtype=np.int32)

    if dihedrals:
        dihedrals_to_write = np.array(dihedral_angles, dtype=np.int32)
    else:
        dihedrals_to_write = np.empty([0, 4], dtype=np.int32)

    mol.atomtype = np.array(atom_types) #pyright: ignore[reportAttributeAccessIssue]
    # The mass of carbon used here is the from OpenMM/AMBER-14 value
    mol.masses = np.repeat(np.array(12.01, dtype=np.float32), len(mol.atomtype)) #pyright: ignore[reportAttributeAccessIssue]
    mol.box = np.zeros((3, 0), dtype=np.float32)
    mol.bonds = bonds_to_write
    mol.angles = angles_to_write #pyright: ignore[reportAttributeAccessIssue]
    mol.dihedrals = dihedrals_to_write #pyright: ignore[reportAttributeAccessIssue]
    if psf_name_out is not None:
        mol.write(psf_name_out)
    return mol
