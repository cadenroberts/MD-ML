import torch
import math
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class Envelope(nn.Module):
    """
    Smooth polynomial envelope function that ensures basis functions decay to zero at cutoff.
    
    This implementation follows PyTorch Geometric's implementation, which uses the form:
    f_env(x) = (1/x + ax^(p-1) + bx^p + cx^(p+1)) for x < 1, and 0 otherwise,
    where x = r/R_c is the normalized distance.
    
    Note: This differs from the original DimeNet paper's formula, which uses:
    f_env(x) = 1 - ((p+1)(p+2)/2)*x^p + p(p+2)*x^(p+1) - (p(p+1)/2)*x^(p+2)
    """
    def __init__(self, exponent: int = 5):
        """
        Initialize envelope function with the specified exponent.
        
        Args:
            exponent (int): Controls the smoothness of decay (default=5, which means p=6)
        """
        super().__init__()
        p = exponent + 1  # e.g. exponent=5 -> p=6
        
        # Pre-compute polynomial coefficients
        self.register_buffer('p', torch.tensor(p, dtype=torch.float32))
        self.register_buffer('a', torch.tensor(-(p+1)*(p+2)/2, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(p*(p+2), dtype=torch.float32))
        self.register_buffer('c', torch.tensor(-p*(p+1)/2, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the envelope function to normalized distances.
        
        Args:
            x (torch.Tensor): Normalized distances (r/R_c) in [0,1]
            
        Returns:
            torch.Tensor: Envelope values, zero beyond cutoff
        """
        p, a, b, c = self.p, self.a, self.b, self.c
        
        # Avoid division by zero at x=0
        # Note: This creates a discontinuity at x=0, but it doesn't affect the
        # final basis function values since they're multiplied by sin(freq*x),
        # which is zero at x=0
        inv_x = torch.where(x > 0, 1.0/x, torch.tensor(0.0, device=x.device))
        
        # Compute polynomial terms
        x_p0 = x.pow(p - 1)        # x^(p-1)
        x_p1 = x_p0 * x            # x^p
        x_p2 = x_p1 * x            # x^(p+1)
        
        # Combine terms - follows PyTorch Geometric implementation
        envelope = (inv_x + a * x_p0 + b * x_p1 + c * x_p2)
        # Zero out beyond cutoff
        envelope = envelope * (x < 1.0).to(x.dtype)
        
        return envelope

class BesselRBF(nn.Module):
    """
    DimeNet-style Bessel radial basis with smooth envelope for distance featurization.
    
    This represents interatomic distances using a set of sine functions modulated
    by a smooth envelope that ensures continuous derivatives at the cutoff.
    """
    def __init__(self, 
                 cutoff_lower: float = 0.0, 
                 cutoff_upper: float = 4.0, 
                 num_rbf: int = 8, 
                 trainable_rbf: bool = False, 
                 dtype: torch.dtype = torch.float32,
                 envelope_exponent: int = 5):
        """
        Initialize the Bessel radial basis functions.
        
        Args:
            cutoff_lower (float): Lower cutoff distance (minimum distance considered)
            cutoff_upper (float): Upper cutoff distance (R_c)
            num_rbf (int): Number of radial basis functions
            trainable (bool): If True, frequencies become trainable parameters
            envelope_exponent (int): Exponent for envelope smoothness (p=exponent+1)
        """
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.envelope = Envelope(envelope_exponent)
        
        # Initialize frequencies ω_n = nπ
        freq_init = torch.arange(1, num_rbf+1, dtype=torch.float32) * math.pi
        if trainable_rbf:
            self.freq = nn.Parameter(freq_init)      # learnable frequencies
        else:
            self.register_buffer('freq', freq_init)  # fixed frequencies

    def reset_parameters(self):
        """Initialize or reset parameters. Required by the TorchMD-Net interface."""
        pass

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Expand interatomic distances into the Bessel basis.
        
        Args:
            distances (torch.Tensor): Tensor of shape (...) containing distances
            
        Returns:
            torch.Tensor: Tensor of shape (..., num_rbf) with basis values for each distance
        """
        # Add channel dimension
        dist = distances.unsqueeze(-1)  # shape (..., 1)
        
        # Normalize distances to [0,1] range
        x = (dist - self.cutoff_lower) / (self.cutoff_upper - self.cutoff_lower)
        
        # Apply envelope and sinusoidal basis: sin(n*pi*x) * envelope(x)
        # Clamp x to avoid issues outside [0, 1] due to numerical precision
        x = torch.clamp(x, 0.0, 1.0)

        # Avoid division by zero in envelope for x=0
        env_val = self.envelope(x)
        
        # Calculate sine term
        sin_term = torch.sin(self.freq * x)
        
        # Combine and handle potential division by zero if distances is exactly zero
        # If dist is 0, x is 0, sin(0)=0, envelope(0)=0, result should be 0
        # The Bessel functions are technically undefined at dist=0, but sin(0)/0 is handled by envelope
        output = env_val * sin_term
        output = torch.where(dist <= self.cutoff_lower, torch.zeros_like(output), output)

        return output


def visualize_basis(rbf_class, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=8, num_points=1000, plot_path=None, title="Basis Functions", **kwargs):
    """
    Visualize the radial basis functions.

    Args:
        rbf_class (nn.Module): The RBF class to visualize (e.g., BesselRBF).
        cutoff_lower (float): Lower cutoff distance.
        cutoff_upper (float): Upper cutoff distance.
        num_rbf (int): Number of basis functions.
        num_points (int): Number of distance points to plot.
        plot_path (str, optional): Path to save the plot. If None, displays the plot.
        title (str): Title for the plot.
        **kwargs: Additional arguments passed to the rbf_class constructor (e.g., envelope_exponent).
    """
    distances = torch.linspace(cutoff_lower, cutoff_upper, num_points)
    
    # Instantiate the RBF layer
    rbf_layer = rbf_class(
        cutoff_lower=cutoff_lower,
        cutoff_upper=cutoff_upper,
        num_rbf=num_rbf,
        **kwargs
    )
    
    # Compute basis function values
    basis_values = rbf_layer(distances)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(num_rbf):
        plt.plot(distances.numpy(), basis_values[:, i].detach().numpy(), label=f'Basis {i+1}')
        
    plt.xlabel('Distance (r)')
    plt.ylabel('Basis Function Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(-1.1, 1.1) # Adjust ylim based on expected RBF range if needed
    plt.xlim(cutoff_lower, cutoff_upper)

    if plot_path:
        plt.savefig(plot_path)
        print(f"Saved basis function plot to {plot_path}")
    else:
        plt.show()


if __name__ == '__main__':
    # Example: Visualize the BesselRBF
    visualize_basis(
        BesselRBF,
        cutoff_upper=4.0,
        num_rbf=8,
        envelope_exponent=5,
        title="Bessel RBF (DimeNet Style)",
        plot_path="bessel_rbf_visualization.png" # Save the plot
    )
