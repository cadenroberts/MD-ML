import re
from typing import Optional, List, Tuple, Dict
import torch
from torch.autograd import grad
from torch import nn, Tensor
from torchmdnet.models import output_modules
from torchmdnet.models.wrappers import AtomFilter
from torchmdnet.models.utils import dtype_mapping
from torchmdnet import priors
from lightning_utilities.core.rank_zero import rank_zero_warn
import warnings
from torchmdnet.models.model import TorchMD_Net


from torchmdnet.models.utils import act_class_mapping

#FIXME: Add license/attribution

class TorchMD_Net_Harmonic(nn.Module):
    """
    An extended TorchMD-Net class that can produce an arbitrary number of outputs in addition to the
    normal scalar + derivative result. Should be drop in compatible with checkpoints for TorchMD_Net.

    Original documentation:
    The TorchMD_Net class combines a given representation model (such as the equivariant transformer),
    an output model (such as the scalar output module), and a prior model (such as the atomref prior).
    It produces a Module that takes as input a series of atom features and outputs a scalar value
    (i.e., energy for each batch/molecule). If `derivative` is True, it also outputs the negative of
    its derivative with respect to the positions (i.e., forces for each atom).

    Parameters
    ----------
    representation_model : nn.Module
        A model that takes as input the atomic numbers, positions, batch indices, and optionally
        charges and spins. It must return a tuple of the form (x, v, z, pos, batch), where x
        are the atom features, v are the vector features (if any), z are the atomic numbers,
        pos are the positions, and batch are the batch indices. See TorchMD_ET for more details.
    output_model : nn.Module
        A model that takes as input the atom features, vector features (if any), atomic numbers,
        positions, and batch indices. See OutputModel for more details.
    prior_model : nn.Module, optional
        A model that takes as input the atom features, atomic numbers, positions, and batch
        indices. See BasePrior for more details. Defaults to None.
    mean : torch.Tensor, optional
        Mean of the training data. Defaults to None.
    std : torch.Tensor, optional
        Standard deviation of the training data. Defaults to None.
    derivative : bool, optional
        Whether to compute the derivative of the outputs via backpropagation. Defaults to False.
    dtype : torch.dtype, optional
        Data type of the model. Defaults to torch.float32.

    """

    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        harmonic_model=None,
        mean=None,
        std=None,
        derivative=False,
        dtype=torch.float32,
        extra_output_models={}
    ):
        super().__init__()
        self.representation_model = representation_model.to(dtype=dtype)
        self.output_model = output_model.to(dtype=dtype)
        self.extra_output_models = torch.nn.ModuleDict(extra_output_models)

        self.harmonic_model = harmonic_model

        assert prior_model is None, "The harmonic model is not compatible with TorchMD-Net style prior models"
        self.prior_model = None

        self.derivative = derivative

        assert mean is None, "The harmonic model does not support data mean normalization"
        assert std is None, "The harmonic model does not support data std normalization"

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean.to(dtype=dtype))
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std.to(dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        # if self.prior_model is not None:
        #     for prior in self.prior_model:
        #         prior.reset_parameters()
        for m in self.extra_output_models.values():
            m.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
        bonds: Optional[Tensor] = None,
        angles: Optional[Tensor] = None,
        dihedrals: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Compute the output of the model.

        This function optionally supports periodic boundary conditions with
        arbitrary triclinic boxes.  The box vectors `a`, `b`, and `c` must satisfy
        certain requirements:

        .. code:: python

           a[1] = a[2] = b[2] = 0
           a[0] >= 2*cutoff, b[1] >= 2*cutoff, c[2] >= 2*cutoff
           a[0] >= 2*b[0]
           a[0] >= 2*c[0]
           b[1] >= 2*c[1]


        These requirements correspond to a particular rotation of the system and
        reduced form of the vectors, as well as the requirement that the cutoff be
        no larger than half the box width.

        Args:
            z (Tensor): Atomic numbers of the atoms in the molecule. Shape: (N,).
            pos (Tensor): Atomic positions in the molecule. Shape: (N, 3).
            batch (Tensor, optional): Batch indices for the atoms in the molecule. Shape: (N,).
            box (Tensor, optional): Box vectors. Shape (3, 3).
            The vectors defining the periodic box.  This must have shape `(3, 3)`,
            where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
            If this is omitted, periodic boundary conditions are not applied.
            q (Tensor, optional): Atomic charges in the molecule. Shape: (N,).
            s (Tensor, optional): Atomic spins in the molecule. Shape: (N,).
            extra_args (Dict[str, Tensor], optional): Extra arguments to pass to the prior model.

        Returns:
            Tuple[Tensor, Optional[Tensor], Dict[str, Tensor]]: Returns:
              * The output of the model
              * The derivative of the output with respect to the positions if derivative is True, or None otherwise.
              * A dict containing the extra_output_models result tensors.
        """
        # Copied what daniel did and removed assertions to allow the model to use precomputed embeddings - Andy
        # assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)
        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(
            z, pos, batch, box=box, q=q, s=s
        )

        # compute extra outputs
        eom_result = {}
        for eom_key, eom in self.extra_output_models.items():
            eom_result[eom_key] = eom.forward(x)

        # Compute harmonic model energy using the representation_model's embeddings
        y_harmonic = None
        if self.harmonic_model is not None:
            y_harmonic, extra_harmonic = self.harmonic_model(x, pos, box, batch, bonds, angles, dihedrals)

            if extra_harmonic:
                eom_result.update(extra_harmonic)

        # apply the output network
        x = self.output_model.pre_reduce(x, v, z, pos, batch)

        # TorchMD_GN + Scalar output converts x (the embeddings) to x (per bead energy) with the above
        # pre_reduce call, then the reduce call below sums up all the energies for a given instance (batch id number).
        #
        # To avoid replicating the rather ugly reduce process I'm going to have harmonic_model output a
        # (rather arbitrary assigned) per bead energy value too and add it here

        if self.harmonic_model is not None:
            x = x + y_harmonic

        # # scale by data standard deviation
        # if self.std is not None:
        #     x = x * self.std

        # aggregate atoms
        x = self.output_model.reduce(x, batch)

        # # shift by data mean
        # if self.mean is not None:
        #     x = x + self.mean

        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        # compute gradients with respect to coordinates
        if self.derivative:
            grad_outputs: List[torch.Tensor] = [torch.ones_like(y)]
            dy = grad(
                [y],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            assert dy is not None, "Autograd returned None for the force prediction."
            return y, -dy, eom_result
        # Returning an empty tensor allows to decorate this method as always returning two tensors.
        # This is required to overcome a TorchScript limitation, xref https://github.com/openmm/openmm-torch/issues/135
        return y, torch.empty(0), eom_result


class HarmonicModel(nn.Module):
    """This class derives the terms for a classical harmonic forcefield from per atom embedding values"""

    def __init__(self,
                 hidden_channels,
                 # FIXME: Enable all terms by default
                 terms=["bonds"],
                #  terms=["bonds", "angles", "dihedrals"],
                 activation="silu",
                 net_shape=[128, 64],
                 return_terms=False,
                 dtype=torch.float,
                 ):
        super().__init__()
        self.terms = terms
        self.hidden_channels = hidden_channels
        self.return_terms = return_terms

        valid_terms = {"bonds", "angles", "dihedrals"}
        assert set(self.terms)-valid_terms == set(), f"Unknown terms: {set(self.terms)-valid_terms}"

        act_class = act_class_mapping[activation]

        self.bond_network = None
        self.angle_network = None
        self.dihedral_network = None

        def build_net(hidden_channels, factor, net_shape):
            result = nn.Sequential(nn.Linear(hidden_channels * factor, net_shape[0], dtype=dtype))
            for i in range(len(net_shape)-1):
                result.append(act_class())
                result.append(nn.Linear(net_shape[i], net_shape[i+1], dtype=dtype))
            return result

        net_shape = net_shape + [2]

        if "bonds" in terms:
            self.bond_network = build_net(hidden_channels, 2, net_shape)

        if "angles" in terms:
            self.angle_network = build_net(hidden_channels, 3, net_shape)

        if "dihedrals" in terms:
            self.dihedral_network = build_net(hidden_channels, 4, net_shape)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bond_network:
            for l in self.bond_network:
                if isinstance(l, nn.Linear):
                    nn.init.xavier_uniform_(l.weight)
                    l.bias.data.fill_(0)

        if self.angle_network:
            for l in self.angle_network:
                if isinstance(l, nn.Linear):
                    nn.init.xavier_uniform_(l.weight)
                    l.bias.data.fill_(0)

        if self.dihedral_network:
            for l in self.dihedral_network:
                if isinstance(l, nn.Linear):
                    nn.init.xavier_uniform_(l.weight)
                    l.bias.data.fill_(0)

    def forward(self, embedding, coords, box, batch, bonds, angles, dihedrals):
        energy = torch.zeros((len(coords),), device=embedding.device)
        extra_output = {}

        box_vectors = None
        if box is not None:
            #FIXME: This assumes the box is always a simple axis aligned rectangular solid!
            box_vectors = box.reshape(-1,9).T[(0,4,8),:].T

        # Bonds
        if self.bond_network:
            a_idx = bonds[:,0]
            b_idx = bonds[:,1]
            vector_list = coords[a_idx]-coords[b_idx]
            if box_vectors is not None:
                # Box wrap vectors (logic from TorchMD)
                all_box = box_vectors[batch[a_idx]]
                vector_list = vector_list - all_box * torch.round(vector_list / all_box)
            dist_list = torch.linalg.norm(vector_list, dim=1)

            # bond_embeddings = torch.hstack([embedding[a_idx],embedding[b_idx]]).reshape(-1, self.hidden_channels*2)
            bond_embeddings = embedding[bonds].reshape(-1, self.hidden_channels*2)
            net_result = self.bond_network(bond_embeddings)

            # Require k0 to be positive
            k0 = net_result[:, 0]**2
            d0 = net_result[:, 1]
            x = dist_list - d0

            per_bond_energy = k0 * (x**2)
            energy.index_add_(0, a_idx, per_bond_energy)

            if self.return_terms:
                extra_output["bond_k0"] = k0
                extra_output["bond_d0"] = d0

        # Angles
        if self.angle_network:
            a1_idx = angles[:,0]
            a2_idx = angles[:,1]
            a3_idx = angles[:,2]
            vec21 = coords[a1_idx] - coords[a2_idx]
            vec23 = coords[a3_idx] - coords[a2_idx]

            if box_vectors is not None:
                # Box wrap vectors (logic from TorchMD)
                all_box = box_vectors[batch[a1_idx]]
                vec21 = vec21 - all_box * torch.round(vec21 / all_box)
                vec23 = vec23 - all_box * torch.round(vec23 / all_box)

            vec21 = vec21 / torch.linalg.norm(vec21, dim=1)[:,None]
            vec23 = vec23 / torch.linalg.norm(vec23, dim=1)[:,None]

            cos_theta = torch.sum(vec21 * vec23, dim=1)
            cos_theta = torch.clamp(cos_theta, -1, 1)
            theta = torch.acos(cos_theta)

            angle_embeddings = embedding[angles].reshape(-1, self.hidden_channels*3)
            net_result = self.angle_network(angle_embeddings)

            # Require k0 to be positive
            k0 = net_result[:, 0]**2
            theta0 = net_result[:, 1]
            # Constrain to valid angles
            theta0 = torch.pi * torch.nn.functional.sigmoid(theta0)
            delta_theta = theta - theta0

            per_angle_energy = k0 * (delta_theta**2)
            energy.index_add_(0, a1_idx, per_angle_energy)

            if self.return_terms:
                extra_output["angle_k0"] = k0
                extra_output["angle_theta0"] = theta0

        if self.dihedral_network:
            # The calculation of phi here was written to match mdtraj's result, but I'm unsure if
            # the phase is the same as TorchMD's dihedral function.
            a1_idx = dihedrals[:,0]
            a2_idx = dihedrals[:,1]
            a3_idx = dihedrals[:,2]
            a4_idx = dihedrals[:,3]
            #FIXME: We should either box wrap these vectors too or never box wrap anything (currently simulate will never wrap within a molecule)
            r12 = coords[a1_idx] - coords[a2_idx]
            r23 = coords[a2_idx] - coords[a3_idx]
            r34 = coords[a3_idx] - coords[a4_idx]
            crossA = torch.cross(r12, r23, dim=1)
            crossB = torch.cross(r23, r34, dim=1)
            crossC = torch.cross(crossA, r23, dim=1)
            normA = torch.norm(crossA, dim=1)
            normB = torch.norm(crossB, dim=1)
            normC = torch.norm(crossC, dim=1)
            normcrossB = crossB / normB.unsqueeze(1)
            cosPhi = torch.sum(crossA * normcrossB, dim=1) / normA
            sinPhi = torch.sum(crossC * normcrossB, dim=1) / normC
            phi = torch.atan2(sinPhi, cosPhi)

            dihedral_embeddings = embedding[dihedrals].reshape(-1, self.hidden_channels*4)
            net_result = self.dihedral_network(dihedral_embeddings)

            # Require k0 to be positive
            k0 = net_result[:, 0]**2
            phi0 = net_result[:, 1]

            # Calculate the periodic distance between phi and phi0
            # https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
            delta_phi = (phi - phi0 + torch.pi) % (2*torch.pi) - torch.pi

            per_dihedral_energy = k0 * (delta_phi**2)
            energy.index_add_(0, a1_idx, per_dihedral_energy)

            if self.return_terms:
                extra_output["dihedral_k0"] = k0
                extra_output["dihedral_phi0"] = phi0

        return energy[:,None], extra_output
