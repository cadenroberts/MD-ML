from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from torchmdnet.models.utils import (
    NeighborEmbedding,
    OptimizedDistance,
    rbf_class_mapping,
    act_class_mapping,
)
import math

from torchmdnet.models.torchmd_gn import InteractionBlock

#FIXME: Add license/attribution

class TorchMD_GN_Ext(nn.Module):
    r"""Graph Network architecture.
        Code adapted from https://github.com/rusty1s/pytorch_geometric/blob/d7d8e5e2edada182d820bbb1eec5f016f50db1e0/torch_geometric/nn/models/schnet.py#L38
        and used at
        Machine learning coarse-grained potentials of protein thermodynamics; M. Majewski et al.
        Nature Communications (2023)

        .. math::
            \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
            h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

        here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
        :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.


    This function optionally supports periodic boundary conditions with arbitrary triclinic boxes.
    For a given cutoff, :math:`r_c`, the box vectors :math:`\vec{a},\vec{b},\vec{c}` must satisfy
    certain requirements:

    .. math::

      \begin{align*}
      a_y = a_z = b_z &= 0 \\
      a_x, b_y, c_z &\geq 2 r_c \\
      a_x &\geq 2  b_x \\
      a_x &\geq 2  c_x \\
      b_y &\geq 2  c_y
      \end{align*}

    These requirements correspond to a particular rotation of the system and reduced form of the vectors, as well as the requirement that the cutoff be no larger than half the box width.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_layers (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        sequence_basis_radius (int, optional): The visible radius of the sequence basis
            function. If 0 sequence information is not used and no basis function wil be
            added.
            (default: :obj:`0`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
        max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
            (default: :obj:`100`)
        max_num_neighbors (int, optional): Maximum number of neighbors to return for a
            given node/atom when constructing the molecular graph during forward passes.
            This attribute is passed to the torch_cluster radius_graph routine keyword
            max_num_neighbors, which normally defaults to 32. Users should set this to
            higher values if they are using higher upper distance cutoffs and expect more
            than 32 neighbors per node/atom. (default: :obj:`32`)
        aggr (str, optional): Aggregation scheme for continuous filter
            convolution ouput. Can be one of 'add', 'mean', or 'max' (see
            https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
            for more details). (default: :obj:`"add"`)
        box_vecs (Tensor, optional):
            The vectors defining the periodic box.  This must have shape `(3, 3)`,
            where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
            If this is omitted, periodic boundary conditions are not applied.
            (default: :obj:`None`)
        check_errors (bool, optional): Whether to check for errors in the distance module.
            (default: :obj:`True`)

    """

    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        external_embedding_channels=0,
        neighbor_embedding=True,
        sequence_basis_radius=0,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=100,
        max_num_neighbors=32,
        check_errors=True,
        aggr="add",
        dtype=torch.float32,
        box_vecs=None,
    ):
        super().__init__()

        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert aggr in [
            "add",
            "mean",
            "max",
        ], 'Argument aggr must be one of: "add", "mean", or "max"'

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.neighbor_embedding = neighbor_embedding
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.aggr = aggr

        act_class = act_class_mapping[activation]

        self.seq_basis_scale = None
        self.sequence_basis_radius = sequence_basis_radius
        if sequence_basis_radius >= 1:
            assert sequence_basis_radius == int(sequence_basis_radius)
            # Select a scale such that the value falls below the cutoff just before the index exceeds the radius
            self.seq_basis_scale = 1/((int(sequence_basis_radius) + 0.9)/math.sqrt(-math.log(0.01)))

        if external_embedding_channels <= 0:
            self.embedding = nn.Embedding(self.max_z, hidden_channels, dtype=dtype)
        else:
            self.external_embedding_channels = external_embedding_channels
            self.embedding = ExternalEmbedding(external_embedding_channels, hidden_channels, activation, dtype=dtype)

        self.distance = OptimizedDistance(
            cutoff_lower,
            cutoff_upper,
            max_num_pairs=-max_num_neighbors,
            box=box_vecs,
            long_edge_index=True,
            check_errors=check_errors,
        )

        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf, dtype=dtype
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels,
                num_rbf,
                cutoff_lower,
                cutoff_upper,
                self.max_z,
                dtype=dtype,
            )
            if neighbor_embedding
            else None
        )

        interaction_input_size = num_rbf
        if self.seq_basis_scale is not None:
            interaction_input_size += 1

        self.interactions = nn.ModuleList()
        for _ in range(num_layers):
            block = InteractionBlock(
                hidden_channels,
                interaction_input_size,
                num_filters,
                act_class,
                cutoff_lower,
                cutoff_upper,
                aggr=self.aggr,
                dtype=dtype,
            )
            self.interactions.append(block)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters() #pyright: ignore[reportAttributeAccessIssue]
        for interaction in self.interactions:
            interaction.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        box: Optional[Tensor] = None,
        s: Optional[Tensor] = None, # We're using s here for the sequence info rather than the spin
        q: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
        x = self.embedding(z)

        edge_index, edge_weight, _ = self.distance(pos, batch, box)
        edge_attr = self.distance_expansion(edge_weight)

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr) #pyright: ignore[reportCallIssue]

        if self.seq_basis_scale is not None:
            assert s is not None # TorchScript cast from Optional[Tensor] -> Tensor
            seq_basis = torch.exp(-((s[edge_index[0]]-s[edge_index[1]])*self.seq_basis_scale)**2)
            # Cut off to zero outside of the requested radius
            seq_basis = torch.where(seq_basis > 0.01, seq_basis, torch.tensor(0.0, dtype=seq_basis.dtype))
            seq_basis = seq_basis.reshape(-1,1)
            edge_attr = torch.cat([edge_attr, seq_basis], dim=1)

        for interaction in self.interactions:
            x = x + interaction(
                x, edge_index, edge_weight, edge_attr, n_atoms=z.shape[0]
            )

        return x, None, z, pos, batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_filters={self.num_filters}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"embedding={self.embedding}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper}, "
            f"aggr={self.aggr}, "
            f"sequence_basis_radius={self.sequence_basis_radius})"
        )

class ExternalEmbedding(nn.Module):
    # A fully connected network that maps an external embedding vector to the hidden_channels dimension
    def __init__(
        self,
        external_embedding_channels,
        hidden_channels,
        activation="silu",
        dtype=torch.float32,
    ):
        super().__init__()

        self.external_embedding_channels = external_embedding_channels
        self.hidden_channels = hidden_channels

        act_class = act_class_mapping[activation]

        # Unsure if this is the best shape for the network... - Daniel
        self.embed_network = torch.nn.Sequential(
            torch.nn.Linear(external_embedding_channels, external_embedding_channels, dtype=dtype),
            act_class(),
            torch.nn.Linear(external_embedding_channels, hidden_channels, dtype=dtype)
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.embed_network:
            for l in self.embed_network:
                if isinstance(l, nn.Linear):
                    nn.init.xavier_uniform_(l.weight)
                    l.bias.data.fill_(0)

    def forward(self, embedding):
        return self.embed_network.forward(embedding)
