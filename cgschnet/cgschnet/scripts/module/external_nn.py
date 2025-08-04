import torch
import numpy as np
from torchmd.parameters import Parameters
# import time

# adapted from https://github.com/torchmd/torchmd-cg/blob/master/torchmd_cg/utils/make_deltaforces.py

class ExternalNN():
    # this class is molecule agnostic. everything molecule specific is in parameters
    def __init__(self, parameters, nnetsBonds, nnetsAngles, nnetsDihedrals, terms, device, precision=torch.float32):
        self.nnetsBonds = nnetsBonds
        self.nnetsAngles = nnetsAngles
        self.nnetsDihedrals = nnetsDihedrals
        self.bondNamesUnq = np.array(list(self.nnetsBonds.keys()))
        self.angleNamesUnq = np.array(list(self.nnetsAngles.keys()))
        self.dihedralNamesUnq = np.array(list(self.nnetsDihedrals.keys()))

        for k in self.nnetsBonds.keys():
            self.nnetsBonds[k]['bestNet'].to(device)

        for k in self.nnetsAngles.keys():
            self.nnetsAngles[k]['bestNet'].to(device)

        for k in self.nnetsDihedrals.keys():
            self.nnetsDihedrals[k]['bestNet'].to(device)
        
        self.device = device
        self.precision = precision
        # self.n_atoms = len(embeddings)
        self.use_box = True

        self.par = parameters
        self.energies = [ene.lower() for ene in terms]

        # Raz: shouldn't really override it here, but I don't have to regen the Mol cache in this case.
        self.par.to_(device)

    def wrap_dist(self, dist, box):
        if box is None or torch.all(box == 0):
            wdist = dist
        else:
            wdist = dist - box * torch.round(dist / box)
        return wdist

    # we need these functions because we now pass a list of boxes that should not be unsqueezed anymore.
    def calculate_distances(self, atom_pos, atom_idx, box):
        assert atom_pos.shape[0] == box.shape[0]
        direction_vec = self.wrap_dist(atom_pos[atom_idx[:, 0]] - atom_pos[atom_idx[:, 1]], box[atom_idx[:, 0]])
        dist = torch.norm(direction_vec, dim=1)
        direction_unitvec = direction_vec / dist.unsqueeze(1)
        return dist, direction_unitvec, direction_vec


    def calculate(self, posSAD, boxesSDD):
        ''' Performs vectorized calculation of the potential energy and forces for all systems in the batch.
            posSAD: torch.Size([nrSystems, nrAtoms, 3]) - letters in suffix denote the dimensions
            boxesSDD: torch.Size([nrSystems, 3, 3]) - letters in suffix denote the dimensions
            '''
        nsystems = posSAD.shape[0]
        potS = torch.zeros(nsystems, dtype=self.precision).to(self.device)
        
        explicit_forces =True

        nrAtoms = posSAD.shape[1]
        forcesSAD = torch.zeros_like(posSAD)
        bondNamesMol = self.par.bond_params['names']
        angleNames = self.par.angle_params['typesXCX']
        
        otherparams = {}
        
        # linearize the positions. posAD is atoms x 3, but all atoms from all systems are not put in one single dimension A. 
        posAD = posSAD.reshape(-1, 3) 
        
        # spos = pos[i]
        sboxes = [b[torch.eye(3).bool()] for b in boxesSDD]   # Use only the diagonal
        
        # repeat the box for each atom
        sboxesSAD = [b.repeat(nrAtoms, 1) for b in sboxes]
        # concatenate the list of tensors into a single tensor
        sboxAD = torch.cat(sboxesSAD, dim=0)
        assert posAD.shape[0] == nrAtoms * nsystems
        assert sboxAD.shape[0] == nrAtoms * nsystems
        # Bonded terms
        if "bonds" in self.energies and self.par.bond_params is not None:
            pairs = self.par.bond_params["idx"]
            assert bondNamesMol.shape[0] == pairs.shape[0]
            nrBondsOneSys = pairs.shape[0] 
            pairsSA2 = [self.par.bond_params["idx"] + i * nrAtoms for i in range(nsystems)]
            pairsA2 = torch.cat(pairsSA2, dim=0)
            assert pairsA2.shape[0] == nrBondsOneSys * nsystems
            
            bond_distB, bond_unitvecB, _ = self.calculate_distances(posAD, pairsA2, sboxAD)
            assert bond_distB.shape[0] == nrBondsOneSys * nsystems
            nrBondsAll = bond_distB.shape[0]
            force_coeffB = torch.zeros((nrBondsAll, 1), dtype=self.precision).to(self.device)
            Eb_all = torch.zeros((nrBondsAll, 1), dtype=self.precision).to(self.device)
            for bondName in list(self.bondNamesUnq): # 0, ('CAE', 'CAD')
                indFiltAll = torch.tensor(bondNamesMol == bondName).repeat(nsystems, 1).reshape(-1)
                if indFiltAll.sum() == 0:
                    continue
                bond_dist_filt = bond_distB[indFiltAll].clone().detach().requires_grad_(True).to(self.device).reshape(-1, 1)
                E = self.nnetsBonds[bondName]['bestNet'](bond_dist_filt)
                # now use backpropagation to compute the forces w.r.t. the bond distance
                force_coeff_filt = torch.autograd.grad(E.sum(), bond_dist_filt, create_graph=True)[0] # take [0] at the end as the function return a singleton tuple (coeffs,)
                
                force_coeffB[indFiltAll] = force_coeff_filt
                
                Eb_all[indFiltAll] = E
            
            potS += Eb_all.reshape((nsystems, nrBondsOneSys)).sum(dim=1)


            otherparams['bond_dist'] = [bond_distB[nrBondsOneSys * s : nrBondsOneSys * (s+1)] for s in range(nsystems)] 
            otherparams['Eb'] = [Eb_all[nrBondsOneSys * s : nrBondsOneSys * (s+1)] for s in range(nsystems)]
            
            # there should be no non-zero force coeffs, otherwise the neural nets are missing some bonds .. turn it off only for running test_deltaforces_nn.py
            
            if explicit_forces:
                assert bond_unitvecB.shape[0] == force_coeffB.shape[0]
                assert bond_unitvecB.shape[1] == 3
                forcevecB = bond_unitvecB * force_coeffB[:, :] # forcevec torch.Size([nrBonds, 3])
                forcesAD = torch.zeros_like(posAD)
                forcesAD.index_add_(0, pairsA2[:, 0], -forcevecB)
                forcesAD.index_add_(0, pairsA2[:, 1], forcevecB)

                forcesSAD += forcesAD.reshape(nsystems, nrAtoms, 3)
                
                
        if "angles" in self.energies and self.par.angle_params is not None:
            nrAnglesOneSys = self.par.angle_params["idx"].shape[0]
            angleIdxAn = self.par.angle_params["idx"].repeat(nsystems, 1) # An suffix denotes  it's a tensor over angles (not atoms!)
            assert angleNames.shape[0] == nrAnglesOneSys

            _, _, ra21 = self.calculate_distances(posAD, angleIdxAn[:, [0, 1]], sboxAD)
            _, _, ra23 = self.calculate_distances(posAD, angleIdxAn[:, [2, 1]], sboxAD)
            
            nrAnglesAll = angleIdxAn.shape[0] # nr of angles over all systems
            assert nrAnglesAll == nrAnglesOneSys * nsystems
            grad_ra21 = torch.zeros((nrAnglesAll, 3), dtype=self.precision).to(self.device)
            grad_ra23 = torch.zeros((nrAnglesAll, 3), dtype=self.precision).to(self.device)
            thetaAll = torch.zeros((nrAnglesAll, 1), dtype=self.precision).to(self.device)
            Eall = torch.zeros((nrAnglesAll, 1), dtype=self.precision).to(self.device)
            for angleName in list(self.angleNamesUnq): # 0, ('CAE', 'CAD')
                indFiltAll = torch.tensor(angleNames == angleName).repeat(nsystems, 1).reshape(-1)
                if indFiltAll.sum() == 0:
                    continue
                
                assert indFiltAll.shape[0] == nrAnglesAll
                ra21Filt = ra21[indFiltAll,:].clone().detach().requires_grad_(True).to(self.device)
                ra23Filt = ra23[indFiltAll,:].clone().detach().requires_grad_(True).to(self.device)
                
                dotprod = torch.sum(ra23Filt * ra21Filt, dim=1)
                norm23inv = 1 / torch.norm(ra23Filt, dim=1)
                norm21inv = 1 / torch.norm(ra21Filt, dim=1)
                
                cos_theta = dotprod * norm21inv * norm23inv
                cos_theta = torch.clamp(cos_theta, -1, 1)
                theta = torch.acos(cos_theta).to(self.precision).reshape(-1,1)
                assert torch.abs(theta).max() < np.pi, 'Theta angles should be in radians'

                E = self.nnetsAngles[angleName]['bestNet'](theta)
                # now use backpropagation to compute the forces w.r.t. the bond distance
                grad_ra21_filt, grad_ra23_filt = torch.autograd.grad(E.sum(), [ra21Filt, ra23Filt], create_graph=True) 
                grad_ra21[indFiltAll] = grad_ra21_filt
                grad_ra23[indFiltAll] = grad_ra23_filt
                

                thetaAll[indFiltAll] = theta # assemble all angles and energies for logging them
                Eall[indFiltAll] = E

            # pot[i] += E.sum()
            potS += Eall.reshape((nsystems, nrAnglesOneSys)).sum(dim=1)

            # now compute the angle forces from the gradients
            angle_force0 = -grad_ra21  # Atom 0 experiences the opposite of r21's gradient
            angle_force2 = -grad_ra23   # Atom 2 experiences r23's gradient
            
            # Atom 1 forces are distributed based on r21, r23
            angle_force1 = grad_ra21 + grad_ra23  # Atom 1 experiences contributions from r21 and r23

            otherparams['ra21'] = [ra21[nrAnglesOneSys * s : nrAnglesOneSys * (s+1)] for s in range(nsystems)]
            otherparams['ra23'] = [ra23[nrAnglesOneSys * s : nrAnglesOneSys * (s+1)] for s in range(nsystems)]
            otherparams['theta'] = [thetaAll[nrAnglesOneSys * s : nrAnglesOneSys * (s+1)] for s in range(nsystems)]
            otherparams['Ea'] = [Eall[nrAnglesOneSys * s : nrAnglesOneSys * (s+1)] for s in range(nsystems)]       
            # otherparams['angle_forces'] = (angle_force0, angle_force1, angle_force2)

            if explicit_forces:
                forcesAD = torch.zeros_like(posAD)
                forcesAD.index_add_(0, angleIdxAn[:, 0], angle_force0)
                forcesAD.index_add_(0, angleIdxAn[:, 1], angle_force1)
                forcesAD.index_add_(0, angleIdxAn[:, 2], angle_force2)
                forcesSAD += forcesAD.reshape(nsystems, nrAtoms, 3)


        if "dihedrals" in self.energies and self.par.dihedral_params is not None:
            nrDihedralsOneSys = self.par.dihedral_params["idx"].shape[0]
            dihed_idx = self.par.dihedral_params["idx"].repeat(nsystems, 1)
            _, _, r12 = self.calculate_distances(posAD, dihed_idx[:, [0, 1]], sboxAD)
            _, _, r23 = self.calculate_distances(posAD, dihed_idx[:, [1, 2]], sboxAD)
            _, _, r34 = self.calculate_distances(posAD, dihed_idx[:, [2, 3]], sboxAD)              

            r12.requires_grad = True
            r23.requires_grad = True
            r34.requires_grad = True

            otherparams['r12'] = [r12[nrDihedralsOneSys * s : nrDihedralsOneSys * (s+1)] for s in range(nsystems)]
            otherparams['r23'] = [r23[nrDihedralsOneSys * s : nrDihedralsOneSys * (s+1)] for s in range(nsystems)]
            otherparams['r34'] = [r34[nrDihedralsOneSys * s : nrDihedralsOneSys * (s+1)] for s in range(nsystems)]
            otherparams['dih_idx'] = [dihed_idx[nrDihedralsOneSys * s : nrDihedralsOneSys * (s+1)] for s in range(nsystems)]
            
            crossA = torch.cross(r12, r23, dim=1)
            crossB = torch.cross(r23, r34, dim=1)
            crossC = torch.cross(r23, crossA, dim=1)
            normA = torch.norm(crossA, dim=1)
            normB = torch.norm(crossB, dim=1)
            normC = torch.norm(crossC, dim=1)
            normcrossB = crossB / normB.unsqueeze(1)
            cosPhi = torch.sum(crossA * normcrossB, dim=1) / normA
            sinPhi = torch.sum(crossC * normcrossB, dim=1) / normC
            phi = -torch.atan2(sinPhi, cosPhi).to(self.precision).reshape(-1,1) # dihedral angle
            
            E = self.nnetsDihedrals['(X, X, X, X)']['bestNet'](phi)
            # pot[i] += E.sum()
            
            grad_r12, grad_r23, grad_r34 = torch.autograd.grad(E.sum(), [r12, r23, r34], create_graph=True)
            
            # Convert the gradients to forces acting on the 4 atoms of the dihedral angle
            force0 = -grad_r12  # Atom 0 experiences the opposite of r12's gradient
            force3 = grad_r34   # Atom 3 experiences r34's gradient

            # Atom 1 and 2 forces are distributed based on r12, r23, and r34
            force1 = grad_r12 - grad_r23  # Atom 1 experiences contributions from r12 and r23
            force2 = grad_r23 - grad_r34  # Atom 2 experiences contributions from r23 and r34

            # Combine the forces into a tuple to match the expected format
            dihedral_forces = (force0, force1, force2, force3)

            # otherparams['dihedral_forces'] = dihedral_forces
            otherparams['E'] = [E[nrDihedralsOneSys * s : nrDihedralsOneSys * (s+1)] for s in range(nsystems)]
            otherparams['phi'] = [phi[nrDihedralsOneSys * s : nrDihedralsOneSys * (s+1)] for s in range(nsystems)]

            potS += E.reshape((nsystems, nrDihedralsOneSys)).sum(dim=1)
            
            if explicit_forces:
                forcesAD = torch.zeros_like(posAD)
                forcesAD.index_add_(0, dihed_idx[:, 0], dihedral_forces[0])
                forcesAD.index_add_(0, dihed_idx[:, 1], dihedral_forces[1])
                forcesAD.index_add_(0, dihed_idx[:, 2], dihedral_forces[2])
                forcesAD.index_add_(0, dihed_idx[:, 3], dihedral_forces[3])
                forcesSAD += forcesAD.reshape(nsystems, nrAtoms, 3)

        return potS.detach(), forcesSAD.detach(), otherparams


class ParametersNN(Parameters):
    def __init__(
        self,
        mol,
        terms,
        precision=torch.float32,
        device="cuda",
    ):

        self.nonbonded_params = None
        self.charges = None
        self.masses = None
        self.mapped_atom_types = None
        self.nonbonded_14_params = None
        self.improper_params = None
        self.natoms = mol.numAtoms
        # if terms is None:
        #     terms = ("bonds", "angles", "dihedrals", "impropers", "1-4", "lj")
        terms = [term.lower() for term in terms]
        
        self.bond_params = self.make_bonds(mol)
        self.angle_params = self.make_angles(mol)
        self.dihedral_params = self.make_dihedrals(mol)            
        
        self.precision_(precision)
        self.to_(device)


    def make_bonds(self, mol): #pyright: ignore[reportIncompatibleMethodOverride]
        bondList = np.unique([sorted(bb) for bb in mol.bonds], axis=0)
        bonds = {}
        bonds["idx"] = torch.tensor(bondList.astype(np.int64)) # [[0,1], [1,2], [2,3], ... ]
        bonds['names'] = np.empty(len(bondList), dtype='U10')
        for i, bb in enumerate(bondList):
            at_t = tuple(mol.atomtype[bb]) # at_t = ('CAP', 'CAH') or ('CAP', 'CAH') or ...
            # since CAT-CAL and CAL-CAT are the same bond, sort alphabetically to have unique names. 
            st = sorted([at_t[0], at_t[1]])
            bonds['names'][i] = '%s-%s' % (st[0], st[1])
        
        return bonds

    def make_angles(self, mol): #pyright: ignore[reportIncompatibleMethodOverride]
        angleList = np.unique(
            [ang if ang[0] < ang[2] else ang[::-1] for ang in mol.angles], axis=0
        )
        angles = {}
        angles["idx"] = torch.tensor(angleList.astype(np.int64))
        angles["typesXCX"] = np.empty(len(angleList), dtype='U20')
        
        for i, ang in enumerate(angleList):
            at_t = tuple(mol.atomtype[ang]) # at_t = ('CAP', 'CAH') or ('CAP', 'CAH') or ...
            angles["typesXCX"][i] = '(X, %s, X)' % (at_t[1])

        return angles

    def make_dihedrals(self, mol): #pyright: ignore[reportIncompatibleMethodOverride]
        uqdihedrals = np.unique(
            [dih if dih[0] < dih[3] else dih[::-1] for dih in mol.dihedrals], axis=0
        )
        dihedrals: dict[str, list | torch.Tensor] = {"map": [], "params": []}
        dihedrals["idx"] = torch.tensor(uqdihedrals.astype(np.int64))

        return dihedrals

    def to_(self, device):
        self.bond_params["idx"] = self.bond_params["idx"].to(device)
        self.angle_params['idx'] = self.angle_params['idx'].to(device)
        self.dihedral_params["idx"] = self.dihedral_params["idx"].to(device) #pyright: ignore[reportAttributeAccessIssue]
        
    def precision_(self, precision):
        pass
        
