import torch
import torch.nn as nn

import yaml
import numpy as np

# Key utility functions
def parse_ordered_key(key_str):
    """Turn a string like '(CAA, CAA)' into a tuple"""
    # Note: This version does not sort the result, so (A, B) != (B, A)
    if key_str.startswith("("):
        key_str = key_str[1:-1]
    return tuple(key_str.split(", "))

def key_to_str(key):
    """Convert a tuple key back into a string"""
    # Single value keys (lj terms) don't have brackets
    if len(key) == 1:
        return str(key)
    return "(" + ", ".join(key) + ")"

def filter_from_keys(keys):
    """Scans a list of key strings and returns corresponding wildcard mask"""
    masks = []
    for k in keys:
        masks.append([i=='X' for i in parse_ordered_key(k)])
    # Assert that there's only one mask pattern in the key set
    assert np.all(np.any(np.array(masks),axis=0) == np.all(np.array(masks),axis=0))
    return np.all(np.array(masks),axis=0).tolist()

def mask_key(mask, key):
    """Apply a wildcard mask to a key tuple"""
    result = []
    for ki, mi in zip(key, mask):
        if mi:
            result.append("X")
        else:
            result.append(ki)
    return tuple(result)

# TorchForceField
# Most of the math here came from TorchMD's forces.py and parameters.py

class TFF_Term(nn.Module):
    def __init__(self):
        super().__init__()

    def make_param_idx(self, prm, mol, term_str, term_idx, device):
        """Generate two mappings:
        * self.term_mapping: Maps each term key in the forcefield definition to
                             an element of the parameters array.
        * self.param_idx: Maps each realized term in the molecule (e.g. the bond
                          between the 1st and 2nd atom) to the corresponding element
                          of the parameters array.
        """
        # Generate a mapping from an atom index to it's type name
        uqatomtypes, indexes = np.unique(mol.atomtype, return_inverse=True)
        # Generate a unique index for each term
        self.term_mapping = {j:i for i,j in enumerate(sorted(prm[term_str].keys()))}

        param_idx = []
        key_filter = filter_from_keys(prm[term_str].keys())
        for b in term_idx:
            key = key_to_str(mask_key(key_filter, tuple(uqatomtypes[indexes[b]])))
            if key not in self.term_mapping:
                key = key_to_str(mask_key(key_filter, tuple(reversed(uqatomtypes[indexes[b]]))))
            param_idx.append(self.term_mapping[key])

        self.param_idx = torch.as_tensor(param_idx, dtype=torch.long, device=device)

class TFF_Bond(TFF_Term):
    def __init__(self, prm, mol, device):
        super().__init__()

        self.make_param_idx(prm, mol, "bonds", mol.bonds, device)

        self.term_params = torch.zeros((len(self.term_mapping),2), dtype=torch.float)
        for k, v in self.term_mapping.items():
            param = prm["bonds"][k]
            self.term_params[v] = torch.as_tensor([param["k0"], param["req"]], dtype=torch.float)
        self.term_params = self.term_params.to(device)

        # The array cast is to convince pytorch that the uint values can safely be converted to longs
        self.coord_idx = torch.as_tensor(np.array(mol.bonds, dtype=int), dtype=torch.long, device=device)
        # Use self.param_idx to assign the params to their corresponding bonds
        self.params = self.term_params[self.param_idx]

    def forward(self, dist_mat, vector_mat, forces_out, calc_energy, calc_forces):
        """Calculate bond forces and add them to forces_out"""
        # Math from TorchMD
        if not len(self.coord_idx):
            return 0.0

        a_idx = self.coord_idx[:,0]
        b_idx = self.coord_idx[:,1]
        bond_dists = dist_mat[a_idx, b_idx]
        bond_vecs = vector_mat[a_idx, b_idx]
        k0 = self.params[:, 0]
        d0 = self.params[:, 1]
        x = bond_dists - d0

        if calc_energy:
            energy = torch.sum(k0 * (x**2))
        else:
            energy = torch.zeros([1], device=forces_out.device)

        if calc_forces:
            bond_force = 2 * k0 * x
            forces_out.index_add_(0, a_idx,  bond_force[:,None]*bond_vecs)
            forces_out.index_add_(0, b_idx, -bond_force[:,None]*bond_vecs)

        return energy

class TFF_Angle(TFF_Term):
    def __init__(self, prm, mol, device):
        super().__init__()

        self.make_param_idx(prm, mol, "angles", mol.angles, device)

        self.term_params = torch.zeros((len(self.term_mapping),2), dtype=torch.float)
        for k, v in self.term_mapping.items():
            param = prm["angles"][k]
            # Phase is originally in degrees, convert to radians
            # k0 is in energy/radians^2
            self.term_params[v] = torch.as_tensor([param["k0"], param["theta0"]*np.pi/180.0], dtype=torch.float)
        self.term_params = self.term_params.to(device)

        # The array cast is to convince pytorch that the uint values can safely be converted to longs
        self.coord_idx = torch.as_tensor(np.array(mol.angles, dtype=int), dtype=torch.long, device=device)
        # Use self.param_idx to assign the params to their corresponding angles
        self.params = self.term_params.to(device)[self.param_idx]

    def forward(self, dist_mat, vector_mat, forces_out, calc_energy, calc_forces):
        """Calculate angle forces and add them to forces_out"""
        if not len(self.coord_idx):
            return 0.0

        # Math from TorchMD
        k0 = self.params[:, 0]
        theta0 = self.params[:, 1]

        a1_idx = self.coord_idx[:,0]
        a2_idx = self.coord_idx[:,1]
        a3_idx = self.coord_idx[:,2]
        vec21 = vector_mat[a2_idx, a1_idx]
        vec23 = vector_mat[a2_idx, a3_idx]
        dist23 = dist_mat[a2_idx, a3_idx]
        dist21 = dist_mat[a2_idx, a1_idx]

        cos_theta = torch.sum(vec21 * vec23, dim=1)
        cos_theta = torch.clamp(cos_theta, -1, 1)
        theta = torch.acos(cos_theta)
        delta_theta = theta - theta0

        if calc_energy:
            energy = torch.sum(k0 * delta_theta * delta_theta)
        else:
            energy = torch.zeros([1], device=forces_out.device)

        if calc_forces:
            sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
            # Deal with divide by zero
            # Note: Using torch.where to broadcast a value isn't legal in a CUDA graph so instead we allow the
            # division then clean up the nans.
            # coef = torch.where(sin_theta != 0, -2.0 * k0 * delta_theta / sin_theta, torch.tensor(0.0))
            coef = -2.0 * k0 * delta_theta / sin_theta
            coef = coef.nan_to_num(nan=0.0)

            force0 = (coef[:, None] * (cos_theta[:, None] * vec21 - vec23)) / dist21[:, None]
            force2 = (coef[:, None] * (cos_theta[:, None] * vec23 - vec21)) / dist23[:, None]
            force1 = -(force0 + force2)

            forces_out.index_add_(0, a1_idx, force0)
            forces_out.index_add_(0, a2_idx, force1)
            forces_out.index_add_(0, a3_idx, force2)

        return energy

class TFF_Dihedral(TFF_Term):
    def __init__(self, prm, mol, device):
        super().__init__()

        self.make_param_idx(prm, mol, "dihedrals", mol.dihedrals, device)

        num_dihderal_terms = max([len(i["terms"]) for i in prm["dihedrals"].values()])
        self.term_params = torch.zeros((len(prm["dihedrals"]), num_dihderal_terms, 3))
        for k, v in self.term_mapping.items():
            param_row = []
            for p in prm["dihedrals"][k]["terms"]:
                # Phase is originally in degrees, convert to radians
                param_row.append([p["per"], p["phi_k"], p["phase"]*np.pi/180.0])
            # Zero pad if the number of terms is less than the max
            while len(param_row) < num_dihderal_terms:
                param_row.append([0.0,0.0,0.0])
            self.term_params[v] = torch.as_tensor(param_row, dtype=torch.float)

        # The array cast is to convince pytorch that the uint values can safely be converted to longs
        self.coord_idx = torch.as_tensor(np.array(mol.dihedrals, dtype=int), dtype=torch.long, device=device)
        # Use self.param_idx to assign the params to their corresponding dihderals
        self.params = self.term_params.to(device)[self.param_idx]

    def forward(self, dist_mat, vector_mat, forces_out, calc_energy, calc_forces):
        """Calculate dihedral forces and add them to forces_out"""
        if not len(self.coord_idx):
            return 0.0

        # Math from TorchMD
        # Calculate 1 coeff per dihedral
        a1_idx = self.coord_idx[:,0]
        a2_idx = self.coord_idx[:,1]
        a3_idx = self.coord_idx[:,2]
        a4_idx = self.coord_idx[:,3]
        vec12 = vector_mat[a1_idx, a2_idx]
        vec23 = vector_mat[a2_idx, a3_idx]
        vec34 = vector_mat[a3_idx, a4_idx]
        dist12 = dist_mat[a1_idx, a2_idx]
        dist23 = dist_mat[a2_idx, a3_idx]
        dist34 = dist_mat[a3_idx, a4_idx]
        # FIXME: I'm unsure if the lengths matter, lets reproduce them for now
        # r12 = vec12 * dist12[:,None]
        # r23 = vec23 * dist23[:,None]
        # r34 = vec34 * dist34[:,None]
        # FIXME: For some reasons my phi comes out negative relative to the TorchMD values unless I reverse the vectors...
        r12 = vec12 * -dist12[:,None]
        r23 = vec23 * -dist23[:,None]
        r34 = vec34 * -dist34[:,None]
        #
        crossA = torch.cross(r12, r23, dim=1)
        crossB = torch.cross(r23, r34, dim=1)
        crossC = torch.cross(r23, crossA, dim=1)
        normA = torch.norm(crossA, dim=1)
        normB = torch.norm(crossB, dim=1)
        normC = torch.norm(crossC, dim=1)
        normcrossB = crossB / normB.unsqueeze(1)
        cosPhi = torch.sum(crossA * normcrossB, dim=1) / normA
        sinPhi = torch.sum(crossC * normcrossB, dim=1) / normC
        phi = -torch.atan2(sinPhi, cosPhi)

        energy = torch.zeros([1], device=forces_out.device)
        if calc_forces:
            coeff = torch.zeros((len(self.coord_idx),), device=self.coord_idx.device)

        # Iterate over the terms
        for i in range(self.params.shape[1]):
            per = self.params[:,i, 0]
            k0 = self.params[:,i, 1]
            phi0 = self.params[:,i, 2]
            angleDiff = per * phi - phi0
            if calc_energy:
                energy += torch.sum(k0 * (1 + torch.cos(angleDiff)))
            if calc_forces:
                coeff += -per * k0 * torch.sin(angleDiff) #pyright: ignore[reportPossiblyUnboundVariable]


        if calc_forces:
            # Calculate 4 forces per coeff
            force0, force1, force2, force3 = None, None, None, None

            # From TorchMD (who took it from OpenMM)
            normDelta2 = torch.norm(r23, dim=1)
            norm2Delta2 = normDelta2**2
            forceFactor0 = (-coeff * normDelta2) / (normA**2) #pyright: ignore[reportOperatorIssue, reportPossiblyUnboundVariable]
            forceFactor1 = torch.sum(r12 * r23, dim=1) / norm2Delta2
            forceFactor2 = torch.sum(r34 * r23, dim=1) / norm2Delta2
            forceFactor3 = (coeff * normDelta2) / (normB**2) #pyright: ignore[reportPossiblyUnboundVariable]

            force0vec = forceFactor0.unsqueeze(1) * crossA
            force3vec = forceFactor3.unsqueeze(1) * crossB
            s = (
                forceFactor1.unsqueeze(1) * force0vec
                - forceFactor2.unsqueeze(1) * force3vec
            )

            force0 = -force0vec
            force1 = force0vec + s
            force2 = force3vec - s
            force3 = -force3vec

            forces_out.index_add_(0, a1_idx, force0)
            forces_out.index_add_(0, a2_idx, force1)
            forces_out.index_add_(0, a3_idx, force2)
            forces_out.index_add_(0, a4_idx, force3)

        return energy

class TFF_RepulsionCG(TFF_Term):
    def __init__(self, prm, mol, device, cutoff, exclusions):
        super().__init__()

        self.cutoff = cutoff
        if not exclusions:
            self.exclusions = []
        else:
            self.exclusions = [i.lower() for i in exclusions]

        uqatomtypes, indexes = np.unique(mol.atomtype, return_inverse=True)

        repulsion_mapping = {j:i for i,j in enumerate(sorted(prm["lj"].keys()))}
        # The params need to be torch.float64 to match TorchMD's coef calculations
        # TODO: Do they really need to be this accurate?
        repulsion_ff_params = torch.zeros((len(repulsion_mapping),2), dtype=torch.float64)
        for k, v in repulsion_mapping.items():
            param = prm["lj"][k]
            repulsion_ff_params[v] = torch.as_tensor([param["sigma"], param["epsilon"]], dtype=torch.float64)

        # The list of all possible repulsions is the upper triangle of the distance matrix
        repulsions = torch.vstack([*torch.triu_indices(mol.numAtoms, mol.numAtoms, offset=1)]).T

        # Remove exclusions from the repulsions list
        #FIXME: Assert all exclusions are valid
        if self.exclusions:
            repulsions = repulsions.tolist()
            if "bonds" in self.exclusions:
                repulsions = [i for i in repulsions if i not in mol.bonds.tolist()]
            if "angles" in self.exclusions:
                repulsions = [i for i in repulsions if i not in [[i[0],i[-1]] for i in mol.angles]]
            if "1-4" in self.exclusions:
                repulsions = [i for i in repulsions if i not in [[i[0],i[-1]] for i in mol.dihedrals]]
            repulsions = torch.as_tensor(repulsions, dtype=torch.long)

        # TorchMD has two parameter matrices corresponding to every possible repulsion combination which
        # in then queries based on the pair list. Because we're going to process the entire matrix anyways
        # we cache the combined values instead.

        param_idx = []
        for rep in repulsions:
            a, b = uqatomtypes[indexes[rep]]
            param_idx.append([repulsion_mapping[a], repulsion_mapping[b]])
        param_idx = torch.as_tensor(param_idx, dtype=torch.long)

        self.param_idx = param_idx.to(device)
        self.coord_idx = repulsions.to(device)

        # Check that we haven't excluded everything
        if len(self.coord_idx):
            # Fetch sigma and epsilon for each atom pair
            sigma_a, epsilon_a, sigma_b, epsilon_b = repulsion_ff_params[param_idx.flatten()].reshape(len(param_idx),-1).T
            # Apply Lorentz - Berthelot combination rule
            # https://pythoninchemistry.org/sim_and_scat/parameterisation/mixing_rules.html
            sigma = 0.5 * (sigma_a + sigma_b)
            epsilon = torch.sqrt(epsilon_a*epsilon_b)
            self.repulsion_B_coef = epsilon * 4 * sigma**6
            self.repulsion_B_coef = self.repulsion_B_coef.float().to(device)

    def forward(self, dist_mat, vector_mat, forces_out, calc_energy, calc_forces):
        """Calculate repulsioncg forces and add them to forces_out"""
        if not len(self.coord_idx):
            return 0.0

        # Math from TorchMD
        a_idx = self.coord_idx[:,0]
        b_idx = self.coord_idx[:,1]
        lj_dists = dist_mat[a_idx, b_idx]
        coef = self.repulsion_B_coef
        lj_vecs = vector_mat[a_idx, b_idx]

        if calc_energy:
            # FIXME: Should this reuse the power calculation like TorchMD does?
            energy_per = coef * (1/lj_dists)**6
            if self.cutoff is not None:
                 energy_per = energy_per * (lj_dists <= self.cutoff).float()
            energy = torch.sum(energy_per)
        else:
            energy = torch.zeros([1], device=forces_out.device)

        if calc_forces:
            force = (-6 * coef * (1/lj_dists)**7)
            if self.cutoff is not None:
                force = force * (lj_dists <= self.cutoff).float()

            forces_out.index_add_(0, a_idx,  force[:,None]*lj_vecs)
            forces_out.index_add_(0, b_idx, -force[:,None]*lj_vecs)

        return energy

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, exclusions={self.exclusions})"

class TorchForceField(nn.Module):
    def __init__(self, prm, mol, device, cutoff=None, terms=None, exclusions=None, use_box=False):
        super().__init__()

        self.mol = mol
        self.device = device
        self.cutoff = cutoff
        assert terms is not None
        assert exclusions is not None
        self.terms = [i.lower() for i in terms]
        self.exclusions = [i.lower() for i in exclusions]
        self.use_box = use_box

        # TODO: Assert exclusions are valid
        valid_terms = {"bonds", "angles", "dihedrals", "repulsioncg", "lj"}
        assert set(self.terms)-valid_terms == set(), f"Unknown terms: {set(self.terms)-valid_terms}"

        with open(prm, "r") as f:
            self.prm = yaml.load(f, Loader=yaml.FullLoader)

        # For compatibility with the TorchMD Forces object we also retain the masses
        self.par = lambda: None # Make an empty object
        self.par.masses = torch.as_tensor([self.prm["masses"][i] for i in mol.atomtype], dtype=torch.float, device=device)[:, None] #pyright: ignore[reportFunctionMemberAccess]

        self.term_modules = nn.ModuleList()
        if "bonds" in self.terms:
            self.term_modules.append(TFF_Bond(self.prm, self.mol, self.device))
        if "angles" in self.terms:
            self.term_modules.append(TFF_Angle(self.prm, self.mol, self.device))
        if "dihedrals" in self.terms:
            self.term_modules.append(TFF_Dihedral(self.prm, self.mol, self.device))
        if "repulsioncg" in self.terms or  "lj" in self.terms:
            self.term_modules.append(TFF_RepulsionCG(self.prm, self.mol, self.device, self.cutoff, self.exclusions))

    def compute_distances(self, coords, box):
        # Compute the flattened distances for the upper triangle
        tri_a, tri_b = torch.triu_indices(self.mol.numAtoms, self.mol.numAtoms, offset=1, device=self.device)
        vector_list = coords[tri_b]-coords[tri_a]
        if self.use_box:
            # Box wrap vectors (logic from TorchMD)
            vector_list = vector_list - box * torch.round(vector_list / box)
        dist_list = torch.linalg.norm(vector_list,axis=1)
        vector_list /= dist_list[:,None]

        # Fill out the full distance matrix
        # Note: I don't know how efficient this is, I mostly picked the list method to avoid dividing
        #       by zero and having to call nan_to_num. - Daniel
        tri_dist_mat = torch.zeros((self.mol.numAtoms,self.mol.numAtoms), device=self.device)
        tri_dist_mat[tri_a, tri_b] = dist_list
        tri_dist_mat[tri_b, tri_a] = dist_list
        tri_vector_mat = torch.zeros((self.mol.numAtoms,self.mol.numAtoms,3), device=self.device)
        tri_vector_mat[tri_a, tri_b] = vector_list
        tri_vector_mat[tri_b, tri_a] = -vector_list
        return tri_dist_mat, tri_vector_mat

    def forward(self, coords, box, forces_out, calc_energy=True, calc_forces=True):
        """Calculate the total energy & forces corresponding to 'coords' in 'box' and write them to forces_out.
        If one of these value is not needed the calculation can be disabled by passing calc_energy=False or
        calc_forces=False, respectively."""

        assert self.use_box == (box is not None)
        if self.use_box:
            assert box.shape == (3,), f"Invalid box tensor, shape={box.shape}"

        forces_out.zero_()
        energy = torch.zeros((1,), device=forces_out.device)
        dist_mat, vector_mat = self.compute_distances(coords, box)

        for m in self.term_modules:
            energy += m.forward(dist_mat, vector_mat, forces_out, calc_energy, calc_forces)

        if not calc_energy:
            return torch.full((1,), float("nan"), dtype=torch.float, device=forces_out.device)
        return energy

    def compute(self, coords, box, forces_out):
        """Calculate forces and energy for an array of replicate systems coords=(nsystems, natoms, 3) and
           box=(nsystems,3,3), saving save the corresponding forces to forces_out=(nsystems, natoms, 3).
           This function is compatible with TorchMD's compute(...) except that total energy is returned as
           a pytorch tensor instead of a numpy array."""

        if box is not None and torch.all(box == 0):
            box = [None for _ in range(len(coords))]
        else:
            # TorchMD's compute expects the matrix form of the box but we only want the diagonal
            box = box.reshape(-1,9)[:,[0,4,8]]
        
        pots = torch.empty((len(coords),1), dtype=torch.float, device=forces_out.device)
        for i in range(len(coords)):
            pots[i,0] = self.forward(coords[i], box[i], forces_out[i])
        return pots
