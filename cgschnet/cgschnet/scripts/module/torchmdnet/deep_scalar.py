import torch
import torch.nn as nn
from torchmdnet.models.output_modules import OutputModel
from torchmdnet.models.utils import act_class_mapping

class DeepScalar(OutputModel):
    """
    A deeper MLP-based output module for TorchMD.

    Inherits from `OutputModel` so that we have the
    necessary `pre_reduce(...)`, `reduce(...)`, etc.
    """
    def make_mlp(self, in_channels, out_channels, hidden_channels,
             num_hidden_layers=2, activation="silu", dtype=None,  dropout: float = 0.0):
        layers = []
        input_dim = in_channels
        for _ in range(num_hidden_layers):
            linear = nn.Linear(input_dim, hidden_channels)
            if dtype is not None:
                linear = linear.to(dtype=dtype)

            layers.append(linear)
            layers.append(act_class_mapping[activation.lower()]())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_channels

        # final layer
        final_linear = nn.Linear(input_dim, out_channels)
        if dtype is not None:
            final_linear = final_linear.to(dtype=dtype)

        layers.append(final_linear)
        return nn.Sequential(*layers)

    def __init__(
        self,
        hidden_channels,
        activation="silu",
        allow_prior_model=True,
        reduce_op="sum",
        dtype=torch.float,
        num_layers=4,
        dropout=0.0,
        **kwargs
    ):
        super().__init__(allow_prior_model=allow_prior_model, reduce_op=reduce_op)
        custom_hidden_dim = kwargs.get("hidden_dim", hidden_channels // 2)

        self.output_network = self.make_mlp(
            in_channels=hidden_channels,
            out_channels=1,
            hidden_channels=custom_hidden_dim,
            activation=activation,
            num_hidden_layers=num_layers,
            dropout=dropout,
            dtype=dtype,
        )

    def reset_parameters(self):
        """
        Called by TorchMD when re-initializing the model.
        """
        for layer in self.output_network:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        """
        This is called by TorchMD_Net BEFORE the `reduce(...)` call.
        """
        return self.output_network(x)
