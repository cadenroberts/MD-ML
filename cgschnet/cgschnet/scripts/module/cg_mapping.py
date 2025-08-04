import numpy as np
import mdtraj
import warnings
from moleculekit.molecule import Molecule
from aggforce import LinearMap, project_forces #type: ignore

def extend_objects(objects, bonds):
    """Forms either angles (if given a list of bonds) or dihedrals (if given a list of angles)
       by looking for a bond that can extend each object. It makes no assumptions about the
       ordering of elements in either list."""

    result = set() # There will be duplicates

    def sort_key(k):
        if k[0] > k[-1]:
            return tuple(reversed(k))
        return tuple(k)

    # loops thru list of bonds/angles
    for a in objects:
        a = list(a)
        for b in bonds:
            b = list(b)
            if a == b:
                continue
            k = None
            if a[0] == b[-1]:
                k = b[:] + a[1:]
            elif a[0] == b[0]:
                k = list(reversed(b[:])) + a[1:]
            elif a[-1] == b[0]:
                k = a[:] + b[1:]
            elif a[-1] == b[-1]:
                k = b[:] + list(reversed(a))[1:]

            if k is not None:
                # An atom can't occur twice in an object
                if max([sum([i == j for j in k]) for i in k]) > 1:
                    continue
                result.add(sort_key(k))

    return list(result)

class CGMapping:
    def __init__(self, topology, map_def):
        """Generate a CG mapping from an all atom MDTraj topology and a CGMappingDef definition"""
        self.src_idx = []
        self.pos_weights = []
        self.force_weights = []

        self.embeddings = []

        self.bead_atom_names = []
        self.bead_types = []
        self.bead_mass = []

        self.cg_topology = mdtraj.Topology()

        for chain in topology.chains:
            last_backbone_idx = None

            if not any([r.is_protein for r in chain.residues]):
                continue # Skip water/ligand/ion chains

            result_chain = self.cg_topology.add_chain()

            # Initially 0 when we haven't seen a protein residue
            # Becomes 1 with the first protein residue
            # Then 2 if we see a non-protein residue
            # This is done ensure the chain is contiguous while still allowing e.g. an ion at the end of a chain
            chain_protein_mode = 0
            for res in chain.residues:
                if not res.is_protein:
                    if chain_protein_mode == 1:
                        chain_protein_mode = 2
                    continue # Skip non-protein residues
                else:
                    if chain_protein_mode == 0:
                        chain_protein_mode = 1
                    assert chain_protein_mode == 1

                idx_mapping = {a.name: a.index for a in res.atoms}
                bead_mapping = map_def.bead_atom_selection[res.name]

                first_bead_idx = self.cg_topology.n_atoms
                # Determine index of this residue's last backbone bead
                # E.g. if the beads were N-CA-C--N-CA-C the offset would be 2 to make C the last backbone
                backbone_idx = self.cg_topology.n_atoms + map_def.bead_backbone_idx[res.name]

                result_res = self.cg_topology.add_residue(res.name, result_chain)

                # for bead_name, bead_type, bead_mass in zip(map_def.bead_atom_name[res.name], map_def.bead_type[res.name], map_def.bead_mass[res.name]):
                for bead_name in map_def.bead_atom_names[res.name]:
                    self.cg_topology.add_atom(bead_name, mdtraj.element.carbon, result_res)
                self.bead_atom_names.extend(map_def.bead_atom_names[res.name])
                self.bead_types.extend(map_def.bead_types[res.name])
                self.bead_mass.extend(map_def.bead_masses[res.name])
                self.embeddings.extend(map_def.bead_embeddings[res.name])

                for bead in bead_mapping:
                    bead_idx = []
                    for atom in bead:
                        if atom not in idx_mapping:
                            # FIXME: The martini mappings seem to have extra atoms (possibly to handle different naming schemes?)
                            raise RuntimeError(f"Missing atom: {res}, {atom}")
                        else:
                            bead_idx.append(idx_mapping[atom])

                    self.src_idx.append(bead_idx)
                    # FIXME: Should use OpenMM masses not mdtraj's
                    bead_weights = np.array([topology.atom(i).element.mass for i in bead_idx])
                    bead_weights = (bead_weights / np.sum(bead_weights)).tolist()
                    self.pos_weights.append(bead_weights)
                    self.force_weights.append(bead_weights)

                if last_backbone_idx is not None:
                    # Add a backbone bond between the first bead of each residue
                    self.cg_topology.add_bond(self.cg_topology.atom(last_backbone_idx), self.cg_topology.atom(first_bead_idx))
                # Sequential bonds from the backbone to the other beads
                for i in range(len(bead_mapping)-1):
                    self.cg_topology.add_bond(self.cg_topology.atom(first_bead_idx+i), self.cg_topology.atom(first_bead_idx+i+1))
                last_backbone_idx = backbone_idx

    def to_mol(self, bonds=True, angles=True, dihedrals=True):
        """Generate a moleculekit Molecule object for the CG topology"""
        mol = Molecule()

        mol.serial = np.arange(self.cg_topology.n_atoms)+1                                                 #pyright: ignore[reportAttributeAccessIssue]
        mol.segid = np.array([str(a.residue.chain.index) for a in self.cg_topology.atoms], dtype=object)   #pyright: ignore[reportAttributeAccessIssue]
        mol.insertion = np.full((self.cg_topology.n_atoms,), '', dtype=object)                             #pyright: ignore[reportAttributeAccessIssue]
        mol.chain = np.copy(mol.segid) # Previously this was left as ''                                    #pyright: ignore[reportAttributeAccessIssue]
        mol.resid = np.array([a.residue.index+1 for a in self.cg_topology.atoms])                          #pyright: ignore[reportAttributeAccessIssue]
        mol.insertion = np.full((self.cg_topology.n_atoms,), '', dtype=object)                             #pyright: ignore[reportAttributeAccessIssue]

        # Requried to make pdbs write correctly
        mol.occupancy = np.full((self.cg_topology.n_atoms,), 1.0, dtype=np.float32)                       #pyright: ignore[reportAttributeAccessIssue]
        mol.beta = np.full((self.cg_topology.n_atoms,), 0.0, dtype=np.float32)                            #pyright: ignore[reportAttributeAccessIssue]
        mol.record = np.full((self.cg_topology.n_atoms,), 'ATOM', dtype=object)                           #pyright: ignore[reportAttributeAccessIssue]
        mol.altloc = np.full((self.cg_topology.n_atoms,), '', dtype=object)                               #pyright: ignore[reportAttributeAccessIssue]
        mol.element = np.full((self.cg_topology.n_atoms,), 'C', dtype=object)                             #pyright: ignore[reportAttributeAccessIssue]
        mol.formalcharge = np.full((self.cg_topology.n_atoms,), 0, dtype=np.int32)                        #pyright: ignore[reportAttributeAccessIssue]

        # The output psf contains resname=res_abbr, name=CA, atomtype=bead_type
        mol.name = np.array(self.bead_atom_names, dtype=object)                                           #pyright: ignore[reportAttributeAccessIssue]
        mol.atomtype = np.array(self.bead_types, dtype=object)                                            #pyright: ignore[reportAttributeAccessIssue]
        mol.resname = np.array([a.residue.name for a in self.cg_topology.atoms], dtype=object)            #pyright: ignore[reportAttributeAccessIssue]

        mol.charge = np.full((self.cg_topology.n_atoms,), 0)                                              #pyright: ignore[reportAttributeAccessIssue]
        mol.masses = np.array(self.bead_mass)                                                             #pyright: ignore[reportAttributeAccessIssue]

        mol.box = np.zeros((3, 0), dtype=np.float32)

        mol.bonds = np.empty((0,2))
        mol.angles = np.empty((0,3))                                                                      #pyright: ignore[reportAttributeAccessIssue]
        mol.dihedrals = np.empty((0,4))                                                                   #pyright: ignore[reportAttributeAccessIssue]

        bonds_to_write = [[b.atom1.index, b.atom2.index] for b  in self.cg_topology.bonds]
        angles_to_write = extend_objects(bonds_to_write, bonds_to_write)
        angles_to_write.sort(key=lambda x: x[0])
        dihedrals_to_write = extend_objects(bonds_to_write, angles_to_write)
        dihedrals_to_write.sort(key=lambda x: x[0])

        if bonds:
            mol.bonds = np.array(bonds_to_write)
        if angles:
            mol.angles = np.array(angles_to_write)
        if dihedrals:
            mol.dihedrals = np.array(dihedrals_to_write)

        return mol

    def to_mdtraj(self):
        """Generate a MDTraj topology object for the CG topology"""
        return self.cg_topology.copy()

    def cg_forces(self, aa_forces):
        """Map all atom forces to CG forces"""
        return self._do_mapping(aa_forces, self.force_weights)

    def cg_positions(self, aa_positions):
        """Map all atom positions to CG forces"""
        return self._do_mapping(aa_positions, self.pos_weights)

    def _do_mapping(self, aa_input, mapping_weights):
        # Apply a weighted mapping to the aa_input (positions or forces)
        num_beads = len(self.src_idx)
        # For each input atom define the bead index it contributes to
        bead_targets = np.concatenate([[i]*len(self.src_idx[i]) for i in range(num_beads)])
        # Flatten the input indices and weights
        bead_idx = np.concatenate(self.src_idx)
        mapping_weights = np.concatenate(mapping_weights, dtype=np.float32)

        num_frames = len(aa_input)
        # Change the axis ordering to (atom, frame, xyz)
        aa_input = aa_input.swapaxes(0,1)[bead_idx]
        weighted_coords = aa_input*mapping_weights[:,None,None]

        bead_output = np.zeros((num_beads, num_frames, 3), dtype=np.float32)
        np.add.at(bead_output, bead_targets, weighted_coords)
        # Change the axis ordering back to (frame, atom, xyz)
        bead_output = bead_output.swapaxes(0,1)
        return bead_output

    def cg_optimal_forces(self, aa_trajectory, aa_forces):
        # get data
        coords = aa_trajectory.xyz

        # Create coordinate mapping and set up bond constraints from all Hydrogen bonds
        cmap = LinearMap(self.src_idx, n_fg_sites=coords.shape[1])
        bonds_array = np.array([(bond.atom1.index, bond.atom2.index)
                                for bond in aa_trajectory.topology.bonds
                                if bond.atom1.element.symbol == 'H'
                                or bond.atom2.element.symbol == 'H'])
        constraintsUnzip = np.array(bonds_array)
        constraints = {frozenset(v) for v in constraintsUnzip}

        # Basic mapping
        # basic_results = project_forces(
        #     forces=forces,
        #     constrained_inds=constraints,
        #     method=constraint_aware_uni_map,
        #     coords=coords,
        #     coord_map=cmap
        # )

        # Statistically optimal mapping
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=r"Converted [PA] to scipy\.sparse\.csc\.csc_matrix")
            optim_results = project_forces(
                forces=aa_forces,
                constrained_inds=constraints,
                coords=coords,
                coord_map=cmap
            )

        # Select only the forces from the results
        return optim_results['mapped_forces'].astype(np.float32)
