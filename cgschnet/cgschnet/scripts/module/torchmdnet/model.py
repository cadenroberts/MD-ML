from typing import Optional, List, Tuple, Dict
import torch
from torch.autograd import grad
from torch import nn, Tensor
from torchmdnet.models import output_modules
from torchmdnet.models.wrappers import AtomFilter
from torchmdnet.models.utils import dtype_mapping, rbf_class_mapping
from torchmdnet import priors
from lightning_utilities.core.rank_zero import rank_zero_warn
from torchmdnet.models.model import TorchMD_Net
from . import deep_scalar 
from .harmonic_model import TorchMD_Net_Harmonic, HarmonicModel
from module.custom_rbf import BesselRBF

# Register the Bessel RBF type
rbf_class_mapping['bessel'] = BesselRBF

#FIXME: Add license/attribution

def create_model(args, prior_model=None, mean=None, std=None):
    """Create a model from the given arguments.

    Run `torchmd-train --help` for a description of the arguments.

    Args:
        args (dict): Arguments for the model.
        prior_model (nn.Module, optional): Prior model to use. Defaults to None.
        mean (torch.Tensor, optional): Mean of the training data. Defaults to None.
        std (torch.Tensor, optional): Standard deviation of the training data. Defaults to None.

    Returns:
        nn.Module: An instance of the TorchMD_Net model.
    """
    dtype = dtype_mapping[args["precision"]]
    if "box_vecs" not in args:
        args["box_vecs"] = None
    if "check_errors" not in args:
        args["check_errors"] = True
    if "static_shapes" not in args:
        args["static_shapes"] = False
    if "vector_cutoff" not in args:
        args["vector_cutoff"] = False

    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        cutoff_lower=float(args["cutoff_lower"]),
        cutoff_upper=float(args["cutoff_upper"]),
        max_z=args["max_z"],
        check_errors=bool(args["check_errors"]),
        max_num_neighbors=args["max_num_neighbors"],
        box_vecs=(
            torch.tensor(args["box_vecs"], dtype=dtype)
            if args["box_vecs"] is not None
            else None
        ),
        dtype=dtype,
    )

    # representation network
    if args["model"] == "graph-network":
        from torchmdnet.models.torchmd_gn import TorchMD_GN

        is_equivariant = False
        representation_model = TorchMD_GN(
            num_filters=args["embedding_dimension"],
            aggr=args["aggr"],
            neighbor_embedding=args["neighbor_embedding"],
            **shared_args, #pyright: ignore[reportArgumentType]
        )
    # FIXME: The old "edge" name is allowed for backwards compatibility but should be removed
    elif (args["model"] == "graph-network-ext") or (args["model"] == "graph-network-edge"):
        from .torchmd_gn_ext import TorchMD_GN_Ext

        is_equivariant = False
        representation_model = TorchMD_GN_Ext(
            num_filters=args["embedding_dimension"],
            aggr=args["aggr"],
            neighbor_embedding=args["neighbor_embedding"],
            external_embedding_channels=args.get("external_embedding_channels", 0),
            sequence_basis_radius=args.get("sequence_basis_radius", 0),
            **shared_args, #pyright: ignore[reportArgumentType]
        )
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    # atom filter
    if not args["derivative"] and args["atom_filter"] > -1:
        representation_model = AtomFilter(representation_model, args["atom_filter"])
    elif args["atom_filter"] > -1:
        raise ValueError("Derivative and atom filter can't be used together")

    # # prior model
    # if args["prior_model"] and prior_model is None:
    #     # instantiate prior model if it was not passed to create_model (i.e. when loading a model)
    #     prior_model = create_prior_models(args)

    # create output network
    if args["output_model"] == "DeepScalar":
        deep_kwargs = args.get("deep_scalar_kwargs", {})

        output_model = deep_scalar.DeepScalar(
            args["embedding_dimension"],
            activation=args["activation"],
            reduce_op=args["reduce_op"],
            dtype=dtype,
            **deep_kwargs
        )
    else:
        output_prefix = "Equivariant" if is_equivariant else ""
        output_cls = getattr(output_modules, output_prefix + args["output_model"])
        output_model = output_cls(
            args["embedding_dimension"],
            activation=args["activation"],
            reduce_op=args["reduce_op"],
            dtype=dtype,
        )

    network_class = TorchMD_Net
    if args.get("extra_outputs") == True:
        network_class = TorchMD_Net_Ext
    if args.get("external_embedding_channels"):
        network_class = TorchMD_Net_Ext

    if args.get("harmonic_net") == True:
        network_class = TorchMD_Net_Harmonic

        harmonic_model = HarmonicModel(
            hidden_channels=args["embedding_dimension"],
            activation=args["activation"],
            terms=args.get("harmonic_net_terms", ["bonds"]),
            net_shape=args.get("harmonic_net_shape", [128, 64]),
            return_terms=args.get("harmonic_net_return_terms", False),
            dtype=dtype,
        )

        # combine representation and output network
        model = network_class(
            representation_model,
            output_model,
            prior_model=prior_model,
            harmonic_model=harmonic_model,
            mean=mean,
            std=std,
            derivative=args["derivative"],
            dtype=dtype,
        )

    else:
        # combine representation and output network
        model = network_class(
            representation_model,
            output_model,
            prior_model=prior_model,
            mean=mean,
            std=std,
            derivative=args["derivative"],
            dtype=dtype,
        )
    return model

class TorchMD_Net_Ext(nn.Module):
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

        if not output_model.allow_prior_model and prior_model is not None:
            prior_model = None
            rank_zero_warn(
                (
                    "Prior model was given but the output model does "
                    "not allow prior models. Dropping the prior model."
                )
            )
        if isinstance(prior_model, priors.base.BasePrior): #pyright: ignore[reportAttributeAccessIssue]
            prior_model = [prior_model]
        self.prior_model = (
            None
            if prior_model is None
            else torch.nn.ModuleList(prior_model).to(dtype=dtype) #pyright: ignore[reportArgumentType]
        )

        self.derivative = derivative

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean.to(dtype=dtype))
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std.to(dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            for prior in self.prior_model:
                prior.reset_parameters()
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
        # # Removed assertions to allow the model to us precomputed embeddings - Daniel
        # assert z.dim() == 1 and z.dtype == torch.long
        # batch = torch.zeros_like(z) if batch is None else batch
        batch = torch.zeros((len(z),), dtype=torch.long, device=z.device) if batch is None else batch

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

        # apply the output network
        x = self.output_model.pre_reduce(x, v, z, pos, batch)

        # scale by data standard deviation
        if self.std is not None:
            x = x * self.std

        # apply atom-wise prior model
        if self.prior_model is not None:
            for prior in self.prior_model:
                x = prior.pre_reduce(x, z, pos, batch, extra_args)

        # aggregate atoms
        x = self.output_model.reduce(x, batch)

        # shift by data mean
        if self.mean is not None:
            x = x + self.mean

        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        # apply molecular-wise prior model
        if self.prior_model is not None:
            for prior in self.prior_model:
                y = prior.post_reduce(y, z, pos, batch, box, extra_args)
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
