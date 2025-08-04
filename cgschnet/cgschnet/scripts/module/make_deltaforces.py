import torch
import numpy as np
from tqdm import tqdm
# from torchmd.forcefields.forcefield import ForceField
from module.torchmd import tagged_forcefield
from torchmd.forces import Forces
from torchmd.systems import System
from moleculekit.molecule import Molecule
from torchmd.parameters import Parameters
# from simulate import CalcWrapper
import time
from module.external_nn import ExternalNN, ParametersNN

# adapted from https://github.com/torchmd/torchmd-cg/blob/master/torchmd_cg/utils/make_deltaforces.py

class DeltaForces:
    def __init__(self, device, psf, coords_npz, box_npz):
        self.device = torch.device(device)
        self.precision = torch.float32
        self.replicas = 1

        self.mol = Molecule(psf)
        self.natoms = self.mol.numAtoms

        self.coords = np.load(coords_npz)
        self.box = None
        if box_npz:
            self.box = np.load(box_npz)

        self.coords = torch.tensor(self.coords, dtype=self.precision).to(device)

        if self.box is not None:
            # Reshape box to be rectangle, then format to be given to set_box
            linearized = self.box.reshape(-1,9).take([0,4,8],axis=1)
            self.box_full = linearized.reshape(linearized.shape[0], 3, 1)
        else:
            self.box_full = torch.zeros(self.coords.shape[0], 3, 1)

        self.prior_forces = torch.zeros((self.coords.shape[0], self.natoms, 3), dtype=self.precision).to('cpu') # store these on CPU
        self.prior_energies = torch.zeros(self.coords.shape[0], dtype=self.precision).to('cpu')
        self.parameters = None
        

    def computePriorForces(self,
        forcefield,
        exclusions=("bonds"),
        forceterms=["Bonds", "Angles", "RepulsionCG"],
        bar_position=0,frames=None
    ):
        # if forceterms is empty list, then we exit
        if forceterms == []:
            return

        ff = tagged_forcefield.create(self.mol, forcefield)
        parameters = Parameters(ff, self.mol, forceterms, precision=self.precision, device=self.device) #pyright: ignore[reportArgumentType]

        system = System(self.natoms, self.replicas, self.precision, self.device)
        system.set_positions(np.zeros((self.natoms, 3, self.replicas)))
        system.set_velocities(torch.zeros(self.replicas, self.natoms, 3))

        forces = Forces(parameters, terms=forceterms, exclusions=exclusions)
        if frames is None: # if None, then process all frames
            frames = range(0, self.coords.shape[0])

        start_time = time.time()
        for i in tqdm(frames, position=bar_position, dynamic_ncols=True, desc="Delta forces - Classical", leave=(bar_position==0)):
            co = self.coords[i]
            system.set_box(self.box_full[i])
            pot = forces.compute(co.reshape([1, self.natoms, 3]), system.box, system.forces)
            fr = (
                system.forces.detach().cpu().reshape([self.natoms, 3])
            )
            self.prior_forces[i,:,:] += fr
            assert len(pot) == 1
            self.prior_energies[i] += pot[0]
        tqdm.write(f"Time taken for classical forces {time.time() - start_time:.2f}")
        

    def makeAndSaveDeltaForces(self, forces_npz, delta_forces_npz, prior_energy_npz):
        all_forces = np.load(forces_npz)
        prior_forces_npy = np.array(self.prior_forces.detach().cpu())
        delta_forces = all_forces - prior_forces_npy
        np.save(delta_forces_npz, delta_forces)
        np.save(prior_energy_npz, self.prior_energies.detach().cpu())

    def addExternalForces(self, forcefield, nnetsBonds, nnetsAngles, nnetsDihedrals, forceterms, bar_position=0, frames=None):
        # if forceterms is empty list, then we exit
        if forceterms == []:
            return

        parameters = ParametersNN(self.mol, forceterms, precision=self.precision, device=self.device) #pyright: ignore[reportArgumentType]

        # for adding the neural network priors. ExternalNN is molecule-agnostic
        calc = ExternalNN(parameters, nnetsBonds, nnetsAngles, nnetsDihedrals, forceterms, self.device)
        tensorbox = torch.tensor(self.box, dtype=self.precision).to(self.device)

        if frames is None: # if None, then process all frames
            frames = range(0, self.coords.shape[0])

        start_time = time.time()
        pot, forces, _ = calc.calculate(self.coords, tensorbox)
        self.prior_forces += forces.detach().cpu()
        self.prior_energies += sum(pot)
        tqdm.write(f"Time taken for neural network forces {time.time() - start_time:.2f}")
