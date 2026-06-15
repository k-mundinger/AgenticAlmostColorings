# ===========================================================================
# Project:      Hadwiger-Nelson
# File:         models.py
# Description:  All sorts of PyTorch models for the parametric case
# ===========================================================================
import math
import sys
from typing import Optional

import matplotlib.pyplot as plt
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utilities import Sin


class ResMLP(nn.Module):
    """A simple MLP with residual connections."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_hidden_layers: int,
                 n_hidden_units: int,
                 activation: str = "relu",
                 output_scale: float = 1.0,
                 initialization: str = "default",
                 disable_residual_connections: bool = False,
                 **kwargs):
        super().__init__()

        # Set attributes
        self.input_dim = input_dim
        self.disable_residual_connections = disable_residual_connections  # If True, residual connections are disabled
        self.input_layer = nn.Linear(input_dim, n_hidden_units)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden_units, n_hidden_units) for _ in range(n_hidden_layers)])
        self.output_layer = nn.Linear(n_hidden_units, output_dim)
        self.output_scale = output_scale
        self.n_colors = output_dim
        

        # Check validity
        assert initialization == 'default' or 'siren' in initialization, 'Initialization must be default or siren.'

        # Set activation function
        if isinstance(activation, list):
            sys.stdout.write(f"Specified activation list: {activation}.\n")
            activation_list = activation
            assert len(
                activation_list) == n_hidden_layers + 1, "Length of activation list must be equal to number of hidden layers + 1."
        else:
            activation_list = [activation] * (n_hidden_layers + 1)

        activation_mapping = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "sin": Sin,
        }
        self.activations = []
        for activation in activation_list:
            assert activation in activation_mapping.keys(), f"Activation {activation} not implemented."
            self.activations.append(activation_mapping[activation]())

        # Set initialization
        if initialization != "default":
            init_split = initialization.split('_')
            if len(init_split) == 1:
                first_layer_scale = 30.0  # Default value from the SIREN paper
            else:
                first_layer_scale = float(init_split[1])
            layers = [self.input_layer] + [hidden_layer for hidden_layer in self.hidden_layers] + [self.output_layer]
            for idx, module in enumerate(layers):
                fan_in = module.in_features
                scale = first_layer_scale if idx == 0 else None
                self.siren_initilization(module, fan_in=fan_in, first_layer_scale=scale)

    def siren_initilization(self, layer: torch.nn.Module, fan_in: int, first_layer_scale: Optional[float] = None):
        """Apply SIREN initialization to a layer.
         Code adapted from https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py"""
        with torch.no_grad():
            if first_layer_scale is not None:
                # We initialize the first layer differently than the rest
                std = 1. / fan_in  # We do not sample directly using std first_layer_scale because the bias is different
            else:
                std = math.sqrt(6. / fan_in)

            layer.weight.uniform_(-std, std)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.uniform_(-std, std)

            if first_layer_scale is not None:
                layer.weight.mul_(first_layer_scale)

    def forward(self, x):
        out = self.activations[0](self.input_layer(x))

        for idx, layer in enumerate(self.hidden_layers):
            activation = self.activations[idx + 1]
            pre_activation = layer(out)
            if not self.disable_residual_connections:
                out = out + activation(pre_activation)
            else:
                out = activation(pre_activation)

        out = self.output_layer(out)
        return self.output_scale * out
    

class DeepONet(nn.Module):

    """
    Implements DeepONet for the polychromatic number 
    case. 
    """

    def __init__(self,
                 trunk_input_dim: int,
                 branch_input_dim: int,
                 latent_dim: int,
                 output_dim: int,
                 trunk_hidden_layers: int,
                 trunk_hidden_units: int,
                 trunk_activation: str,
                 branch_hidden_layers: int,
                 branch_hidden_units: int,
                 branch_activation: str,
                 **kwargs):

        super().__init__()

        activation_mapping = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "sin": Sin,
        }

        self.trunk_activation = activation_mapping[trunk_activation]()
        self.branch_activation = activation_mapping[branch_activation]()

        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_input_dim, trunk_hidden_units),
            self.trunk_activation,
            *[nn.Sequential(nn.Linear(trunk_hidden_units, trunk_hidden_units), self.trunk_activation) for _ in range(trunk_hidden_layers - 1)],
            nn.Linear(trunk_hidden_units, latent_dim*output_dim),
            self.trunk_activation

        )

        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, branch_hidden_units),
            self.branch_activation,
            *[nn.Sequential(nn.Linear(branch_hidden_units, branch_hidden_units), self.branch_activation) for _ in range(branch_hidden_layers - 1)],
            nn.Linear(branch_hidden_units, latent_dim),
            self.branch_activation
        )

        self.latent_dim = latent_dim
        self.output_dim = output_dim

    def forward(self, x):

        reshape = False
        if len(x.shape) == 3:
            # hack because our code sucks
            reshape = True
            batch_size, n_circle_points, _ = x.shape
            x = x.flatten(0, 1)
   


        x_trunk = x[..., :2]
        x_branch = x[..., 2:]

        z_trunk = self.trunk_net(x_trunk).reshape(-1, self.latent_dim, self.output_dim)
        z_branch = self.branch_net(x_branch)

        # print(f"{x.shape=}\n")
        # print(f"{z_trunk.shape=}\n")
        # print(f"{z_branch.shape=}\n")


        out = torch.einsum('blo, bl -> bo', z_trunk, z_branch)

        if reshape:
            out = out.reshape(batch_size, n_circle_points, -1)

        return out






class VoronoiCenters(nn.Module):
    def __init__(self, 
                 num_centers : int, 
                 num_colors : int, 
                 temperature : float,
                 grid_bound : float,
                 average_probabilities : bool,
                 input_dim : int,
                 initial_centers : Optional[list] = None,
                 initial_colors : Optional[list] = None,
                 initial_color_logit_scale : Optional[float] = None,
                 color_weights : Optional[list] = None,
                 fix_logits : Optional[bool] = False,
                 fix_centers : Optional[bool] = False,
                 device : Optional[str] = 'cuda',
                 **kwargs):
        super().__init__()


        self.device = device
        self.temperature = temperature
        self.average_probabilities = average_probabilities
        self.grid_bound = grid_bound
        self.dim = input_dim
        if color_weights is None:
            self.color_weights = torch.tensor([1.0] * num_colors).to(self.device)
        else:
            self.color_weights = torch.tensor(color_weights).to(self.device)

        if initial_centers is not None:
            assert len(initial_centers) == num_centers, f"Number of initial centers ({len(initial_centers)}) must match num_centers ({num_centers})"
            initial_centers_tensor = torch.tensor(initial_centers, dtype=torch.float32)
            self.center_coords = nn.Parameter(initial_centers_tensor)
            #sys.stdout.write(f"Initial centers have been specified: \n{self.center_coords.T}.\n")
        else:
            # center coords uniformly in [-grid_bound, grid_bound]^dim
            self.center_coords = nn.Parameter(torch.rand(num_centers, self.dim) * 2 * grid_bound - grid_bound)

        # Initialize center_logits
        center_logits = torch.randn(num_centers, num_colors)

        if initial_colors is not None:
            assert len(initial_colors) == num_centers, f"Number of initial colors ({len(initial_colors)}) must match num_centers ({num_centers})"
            assert all(0 <= color < num_colors for color in initial_colors), f"All initial colors must be integers between 0 and {num_colors - 1}"

            center_logits = torch.zeros(num_centers, num_colors)
            
            # Set high logit value for specified colors
            for i, color in enumerate(initial_colors): 
                center_logits[i, color] = initial_color_logit_scale
            
            center_logits = center_logits 
            
            #sys.stdout.write(f"Initial colors have been specified: {initial_colors}\n")
            #sys.stdout.write(f"Initial color logit scale: {initial_color_logit_scale}\n")

        self.center_logits = nn.Parameter(center_logits)

        if fix_logits:
            self.center_logits.requires_grad = False
        if fix_centers:
            self.center_coords.requires_grad = False

        self.center_colours = torch.argmax(self.center_logits, dim=1)

        self.distance_weights = self.color_weights[self.center_colours]

    def forward(self, x):
        # shape of x should be (batch_size, 2) - sometimes it is (batch_size, another_batch_size, 2)
        if len(x.shape) == 3:
            reshape = True
            batch_size, n_circle_points, _ = x.shape
            x = x.flatten(0, 1)
        else:
            reshape = False

  
        if x.shape[-1] > 3: # handling off-diagonal case. TODO: it's a hack since input_dim is already with the concatenated colours
            x = x[..., :2]


        # Voronoi diagram computation
        distances = torch.norm(x[:, None, :] - self.center_coords, dim=2)


        # shape of distances is (batch_size, num_centers) - for each point, the distance to all centers
        inverse_distances = self.distance_weights / distances

        # mean of center logits (or maybe center probabilities?!) weighted by inverse distances

        if self.temperature >= 0:
            if self.average_probabilities:
       
                center_probs = self.center_logits
                color_distribution = torch.softmax(inverse_distances * self.temperature, dim = 1) @ center_probs
            else:
                color_distribution = torch.softmax(inverse_distances @ self.center_logits * self.color_weights.to(self.center_logits.device) * self.temperature, dim = 1)
        elif self.temperature == -1:
            # just take logits of the closest center
            closest_centers = torch.argmax(inverse_distances, dim=1)
            color_distribution = self.center_logits[closest_centers]
        else:
            raise ValueError(f"Invalid temperature {self.temperature}.") 

        if reshape:
            color_distribution = color_distribution.reshape(batch_size, n_circle_points, -1)

        return color_distribution

    def extra_repr(self) -> str:
        return f'num_centers={self.center_coords.shape[0]}, num_colors={self.center_logits.shape[1]}, temperature={self.temperature}, grid_bound={self.grid_bound}, average_probabilities={self.average_probabilities}'
    
    def plot_logits(self):

        plt.figure(figsize=(8, 4))
        num_centers = self.center_logits.shape[0]
        n_colours = self.center_logits.shape[1]

        plt.imshow(self.center_logits.detach().cpu().transpose(0, 1), cmap = plt.cm.rainbow)

        # make a box not a rectangle
        plt.title("Logits")
        plt.gca().set_aspect(num_centers / n_colours / 2, adjustable='box')
        plt.colorbar(shrink = .8)
        plt.show()

    def plot_centers(self):

        colors = torch.argmax(self.center_logits, dim=1).cpu()

        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.scatter(self.center_coords[:, 0].detach().cpu(), self.center_coords[:, 1].detach().cpu(), c=colors.detach().cpu(), cmap=plt.cm.Pastel1, marker='o', edgecolors = 'black', s=100)
        plt.subplot(122)
        plt.scatter(self.center_coords[:, 0].detach().cpu(), self.center_coords[:, 1].detach().cpu(), c=colors.detach().cpu(), cmap=plt.cm.Pastel1, marker='o', edgecolors = 'black', s=100)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)

class ParametrizedVoronoiCenters(nn.Module):

    def __init__(self,
                 alpha: float,
                 beta: float,
                 gamma: float,
                 temperature: float,
                 average_probabilities: bool,
                 device : Optional[str] = 'cuda',
                 **kwargs):
        
        super().__init__()

        self.alpha = torch.nn.Parameter(torch.tensor(alpha))
        self.beta = torch.nn.Parameter(torch.tensor(beta))
        self.gamma = torch.nn.Parameter(torch.tensor(gamma))
        self.temperature = temperature
        self.average_probabilities = average_probabilities

        self.device = device

        self.get_center_coords_and_colours()

    def get_center_coords_and_colours(self):

        nrows = 7
        ncols = 7

        xs = []
        ys = []
        color_list = []

        rows = torch.arange(-nrows, nrows, device=self.device)
        cols = torch.arange(-ncols, ncols, device=self.device)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")

        # Compute x and y
        x = col_grid * self.beta
        y = row_grid * self.alpha + col_grid * self.gamma

        # Compute colors
        # Create a tensor that will store all possible color choices for each `col` parity.
        color_choices = torch.tensor([[0, 1, 2], [3, 4, 5]], device=self.device)

        # Select color choices based on whether `col` is even or odd, and apply modulo operation
        is_even_col = (col_grid % 2 == 0).long()  # Converts to indices: 0 for even, 1 for odd
        color = color_choices[is_even_col, (row_grid + col_grid) % 3]

        # Reshape x, y, and color into vectors for output
        xs = x.flatten()
        ys = y.flatten()
        color_list = color.flatten()

        # Construct the final tensors
        xy_tensor = torch.stack((xs, ys), dim=1).to(self.device)
        logit_tensor = F.one_hot(color_list, 6).float().to(self.device)

        self.center_coords = xy_tensor
        self.center_logits = logit_tensor


    def forward(self, x):

        self.get_center_coords_and_colours()

        if len(x.shape) == 3:
            reshape = True
            batch_size, n_circle_points, _ = x.shape
            x = x.flatten(0, 1)
        else:
            reshape = False

        if x.shape[-1] != 2: # handling off-diagonal case. TODO: add dim parameter for 3D case
            x = x[..., :2]

        # Voronoi diagram computation
        distances = torch.norm(x[:, None, :] - self.center_coords, dim=2)

        # shape of distances is (batch_size, num_centers) - for each point, the distance to all centers
        inverse_distances = 1 / distances

        # mean of center logits (or maybe center probabilities?!) weighted by inverse distances

        if self.temperature >= 0:
            if self.average_probabilities:
                center_probs = torch.softmax(self.center_logits, dim=1)
                color_distribution = torch.softmax(inverse_distances * self.temperature, dim = 1) @ center_probs
            else:
                color_distribution = torch.softmax(inverse_distances @ self.center_logits * self.temperature, dim = 1)
        elif self.temperature == -1:
            # just take logits of the closest center
            closest_centers = torch.argmax(inverse_distances, dim=1)
            color_distribution = self.center_logits[closest_centers]
        else:
            raise ValueError(f"Invalid temperature {self.temperature}.") 

        if reshape:
            color_distribution = color_distribution.reshape(batch_size, n_circle_points, -1)

        return color_distribution

class ParametrizedVoronoiCenters3D(nn.Module):

    def __init__(self,
                 generator: torch.tensor,
                 offsets: torch.tensor,
                 box_size: int,
                 temperature: float,
                 average_probabilities: bool,
                 device : Optional[str] = 'cuda',
                 **kwargs):
        
        super().__init__()

        self.device = device
        self.generator = torch.nn.Parameter(torch.tensor(generator).to(self.device))
        self.offsets = torch.nn.Parameter(torch.tensor(offsets).to(self.device))
        self.box_size = box_size

        color_list = torch.cat([i*torch.ones((2*self.box_size)**3, dtype=torch.long) for i in range(14)], dim=0)
        self.center_logits = F.one_hot(color_list, 14).float().to(self.device)
        
        self.temperature = temperature
        self.average_probabilities = average_probabilities

        print(f"{self.generator.shape=}\n")
        print(f"{self.offsets.shape=}\n")

        self.get_center_coords_and_colours()

        print(f"{self.center_coords.shape=}\n")
        print(f"{self.center_logits.shape=}\n")

    def get_center_coords_and_colours(self):

        grid = torch.stack([i*self.generator[0] + j*self.generator[1] + k*self.generator[2] for i in range(-self.box_size, self.box_size) for j in range(-self.box_size, self.box_size) for k in range(-self.box_size, self.box_size)])
        
        full_grid = torch.cat([grid + offset for offset in self.offsets], dim=0).to(self.device)
        
        self.center_coords = full_grid
        

    def forward(self, x):

        self.get_center_coords_and_colours()

        if len(x.shape) == 3:
            reshape = True
            batch_size, n_circle_points, _ = x.shape
            x = x.flatten(0, 1)
        else:
            reshape = False

        # Voronoi diagram computation
        distances = torch.norm(x[:, None, :] - self.center_coords, dim=2)

        # shape of distances is (batch_size, num_centers) - for each point, the distance to all centers
        inverse_distances = 1 / distances

        # mean of center logits (or maybe center probabilities?!) weighted by inverse distances

        if self.temperature >= 0:
            if self.average_probabilities:
                center_probs = self.center_logits
                color_distribution = torch.softmax(inverse_distances * self.temperature, dim = 1) @ center_probs
            else:
                color_distribution = torch.softmax(inverse_distances @ self.center_logits * self.temperature, dim = 1)
        elif self.temperature == -1:
            # just take logits of the closest center
            closest_centers = torch.argmax(inverse_distances, dim=1)
            color_distribution = self.center_logits[closest_centers]
        else:
            raise ValueError(f"Invalid temperature {self.temperature}.") 

        if reshape:
            color_distribution = color_distribution.reshape(batch_size, n_circle_points, -1)

        return color_distribution
    
class ParametrizedVoronoiCenters2D(nn.Module):

    def __init__(self,
                 generator: torch.tensor,
                 offsets: torch.tensor,
                 box_size: int,
                 temperature: float,
                 average_probabilities: bool,
                 device : Optional[str] = 'cuda',
                 trainable_logits : bool = False,
                 **kwargs):
        
        super().__init__()

        self.device = device
        self.generator = torch.nn.Parameter(torch.tensor(generator).to(self.device))
        self.offsets = torch.nn.Parameter(torch.tensor(offsets).to(self.device))
        self.box_size = box_size

        color_list = torch.cat([i*torch.ones((2*self.box_size)**2, dtype=torch.long) for i in range(6)], dim=0)

        if trainable_logits:
            self.center_logits = torch.nn.Parameter(F.one_hot(color_list, 6).float().to(self.device))
        else:
            self.center_logits = F.one_hot(color_list, 6).float().to(self.device)

        self.temperature = temperature
        self.average_probabilities = average_probabilities

        print(f"{self.generator.shape=}\n")
        print(f"{self.offsets.shape=}\n")

        self.get_center_coords_and_colours()

        print(f"{self.center_coords.shape=}\n")
        print(f"{self.center_logits.shape=}\n")

    def get_center_coords_and_colours(self):

        grid = torch.stack([i*self.generator[0] + j*self.generator[1] for i in range(-self.box_size, self.box_size) for j in range(-self.box_size, self.box_size)])
        
        full_grid = torch.cat([grid + offset for offset in self.offsets], dim=0).to(self.device)
    

        # Construct the final tensors
        #logit_tensor = F.one_hot(color_list, 6).float().to(self.device)

        self.center_coords = full_grid
        #self.center_logits = logit_tensor

    def forward(self, x):

        self.get_center_coords_and_colours()

        if len(x.shape) == 3:
            reshape = True
            batch_size, n_circle_points, _ = x.shape
            x = x.flatten(0, 1)
        else:
            reshape = False

        # Voronoi diagram computation
        distances = torch.norm(x[:, None, :] - self.center_coords, dim=2)

        # shape of distances is (batch_size, num_centers) - for each point, the distance to all centers
        inverse_distances = 1 / distances

        # mean of center logits (or maybe center probabilities?!) weighted by inverse distances

        if self.temperature >= 0:
            if self.average_probabilities:
                #center_probs = torch.softmax(self.center_logits, dim=1)
                # hacky: this could create problems in the setting of trainable logits 
                center_probs = self.center_logits
                color_distribution = torch.softmax(inverse_distances * self.temperature, dim = 1) @ center_probs
            else:
                color_distribution = torch.softmax(inverse_distances @ self.center_logits * self.temperature, dim = 1)
        elif self.temperature == -1:
            # just take logits of the closest center
            closest_centers = torch.argmax(inverse_distances, dim=1)
            color_distribution = self.center_logits[closest_centers]
        else:
            raise ValueError(f"Invalid temperature {self.temperature}.") 

        if reshape:
            color_distribution = color_distribution.reshape(batch_size, n_circle_points, -1)

        return color_distribution

class HexagonalVoronoi(nn.Module):

    def __init__(self,
                 z_dist: float,
                 hex_size: float,
                 n_rows: int,
                 n_cols: int,
                 z_repeat: int,
                 box_size: int,
                 temperature: float,
                 average_probabilities: bool,
                 device : Optional[str] = 'cuda',
                 **kwargs):
        
        super().__init__()

        self.device = device
        self.z_dist = torch.nn.Parameter(torch.tensor([z_dist]).to(self.device))
        self.hex_size = torch.nn.Parameter(torch.tensor([hex_size]).to(self.device))
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.z_repeat = z_repeat
        self.box_size = box_size

        z_offsets =  2*torch.rand((2*self.n_rows + 1)*(2*self.n_cols + 1)*7, 1) - 1
        self.z_offsets = torch.nn.Parameter(z_offsets.to(self.device))
        
        self.temperature = temperature
        self.average_probabilities = average_probabilities


        self.get_center_coords()

        print(f"{self.center_coords.shape=}\n")
        print(f"{self.center_logits.shape=}\n")

    def get_center_coords(self):

        n_rows = 2
        n_cols = 2

        pi = torch.tensor(np.pi).to(self.device)

        first_basis_vector = torch.cat([
            2*self.hex_size*torch.cos(pi / 3) + self.hex_size*torch.cos(2 * pi / 3),
            2*self.hex_size*torch.sin(pi / 3) + self.hex_size*torch.sin(2 * pi / 3)
        ])

        second_basis_vector = torch.cat([
            2*self.hex_size*torch.cos(2*pi / 3) + self.hex_size*torch.cos(3*pi / 3),
            2*self.hex_size*torch.sin(2*pi / 3) + self.hex_size*torch.sin(3*pi / 3)
        ])

        two_d_single_colour_grid = torch.stack([i*first_basis_vector + j*second_basis_vector for i in range(-n_rows, n_rows+1) for j in range(-n_cols, n_cols+1)])

        zero_offset = torch.tensor([[0., 0.]]).to(self.device)  # shape: [1, 2]
        hex_offsets = torch.stack([
            torch.tensor([
                self.hex_size * torch.cos(i*pi/3),
                self.hex_size * torch.sin(i*pi/3)
            ]).to(self.device) for i in range(6)
        ])  # shape: [6, 2]
        xy_offsets = torch.cat([zero_offset, hex_offsets], dim=0) 

        full_2d_grid = torch.cat([two_d_single_colour_grid + xy_offset for xy_offset in xy_offsets])
        colours_2d = torch.tensor([[i]*len(two_d_single_colour_grid) for i in range(7)]).flatten()

        shifted_2d_grid = torch.cat((full_2d_grid.to(self.device), self.z_offsets), dim = -1)

        full_3d_grid = torch.cat([shifted_2d_grid + torch.cat([torch.zeros(shifted_2d_grid.shape[0], 2).to(self.device) , i*self.z_dist*torch.ones(shifted_2d_grid.shape[0]).to(self.device)[:, None]], axis = -1) for i in range(-self.z_repeat, self.z_repeat+3)])

        colours_3d = torch.cat([colours_2d if i % 2 == 0 else colours_2d + 7 for i in range(2*(self.z_repeat + 1) + 1)])

        # Construct the final tensors
        logit_tensor = F.one_hot(colours_3d, 14).float().to(self.device)

        self.center_coords = full_3d_grid
        self.center_logits = logit_tensor

    def forward(self, x):

        self.get_center_coords()

        if len(x.shape) == 3:
            reshape = True
            batch_size, n_circle_points, _ = x.shape
            x = x.flatten(0, 1)
        else:
            reshape = False

        # Voronoi diagram computation
        distances = torch.norm(x[:, None, :] - self.center_coords, dim=2)

        # shape of distances is (batch_size, num_centers) - for each point, the distance to all centers
        inverse_distances = 1 / distances

        # mean of center logits (or maybe center probabilities?!) weighted by inverse distances

        if self.temperature >= 0:
            if self.average_probabilities:
                center_probs = self.center_logits
                color_distribution = torch.softmax(inverse_distances * self.temperature, dim = 1) @ center_probs
            else:
                color_distribution = torch.softmax(inverse_distances @ self.center_logits * self.temperature, dim = 1)
        elif self.temperature == -1:
            # just take logits of the closest center
            closest_centers = torch.argmax(inverse_distances, dim=1)
            color_distribution = self.center_logits[closest_centers]
        else:
            raise ValueError(f"Invalid temperature {self.temperature}.") 

        if reshape:
            color_distribution = color_distribution.reshape(batch_size, n_circle_points, -1)

        return color_distribution
    

class LowDimHexagonalVoronoi(nn.Module):

    """
    In this implementation, not every 'column' in the
    xy-plane has its own z-offset. Instead, we have 9
    parameters giving us the relative z-offsets to the 
    neighboring color centers.
    """

    def __init__(self,
                 z_dist: float,
                 hex_size: float,
                 n_rows: int,
                 n_cols: int,
                 z_repeat: int,
                 purple_orange,
                 orange_pink,
                 orange_blue,
                 pink_blue,
                 red_yellow,
                 yellow_purple,
                 red_orange,
                 green_blue,
                 yellow_pink,
                 box_size: int,
                 temperature: float,
                 average_probabilities: bool,
                 device : Optional[str] = 'cuda',
                 **kwargs):
        
        super().__init__()

        self.device = device
        self.z_dist = torch.nn.Parameter(torch.tensor([z_dist]).to(self.device))
        self.hex_size = torch.nn.Parameter(torch.tensor([hex_size]).to(self.device))
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.z_repeat = z_repeat
        self.box_size = box_size

        self.trainable_offsets = torch.nn.ParameterDict({
            'purple_orange': torch.nn.Parameter(torch.tensor(purple_orange).to(self.device)),
            'orange_pink': torch.nn.Parameter(torch.tensor(orange_pink).to(self.device)),
            'orange_blue': torch.nn.Parameter(torch.tensor(orange_blue).to(self.device)),
            'pink_blue': torch.nn.Parameter(torch.tensor(pink_blue).to(self.device)),
            'red_yellow': torch.nn.Parameter(torch.tensor(red_yellow).to(self.device)),
            'yellow_purple': torch.nn.Parameter(torch.tensor(yellow_purple).to(self.device)),
            'red_orange': torch.nn.Parameter(torch.tensor(red_orange).to(self.device)),
            'green_blue': torch.nn.Parameter(torch.tensor(green_blue).to(self.device)),
            'yellow_pink': torch.nn.Parameter(torch.tensor(yellow_pink).to(self.device))
        })
        
        self.temperature = temperature
        self.average_probabilities = average_probabilities

        _, initial_full_2d_grid = self.get_full_2d_grid()
        self.hex_neighbors = self.get_hex_neighbors(initial_full_2d_grid, self.n_rows, self.n_cols)


        self.get_center_coords()

        print(f"{self.center_coords.shape=}\n")
        print(f"{self.center_logits.shape=}\n")

    def get_full_2d_grid(self):

        

        pi = torch.tensor(np.pi).to(self.device)

        first_basis_vector = torch.cat([
            2*self.hex_size*torch.cos(pi / 3) + self.hex_size*torch.cos(2 * pi / 3),
            2*self.hex_size*torch.sin(pi / 3) + self.hex_size*torch.sin(2 * pi / 3)
        ])

        second_basis_vector = torch.cat([
            2*self.hex_size*torch.cos(2*pi / 3) + self.hex_size*torch.cos(3*pi / 3),
            2*self.hex_size*torch.sin(2*pi / 3) + self.hex_size*torch.sin(3*pi / 3)
        ])

        two_d_single_colour_grid = torch.stack([i*first_basis_vector + j*second_basis_vector for i in range(-self.n_rows, self.n_rows+1) for j in range(-self.n_cols, self.n_cols+1)])

        zero_offset = torch.tensor([[0., 0.]]).to(self.device)  # shape: [1, 2]
        hex_offsets = torch.stack([
            torch.tensor([
                self.hex_size * torch.cos(i*pi/3),
                self.hex_size * torch.sin(i*pi/3)
            ]).to(self.device) for i in range(6)
        ])  # shape: [6, 2]
        xy_offsets = torch.cat([zero_offset, hex_offsets], dim=0) 

        full_2d_grid = torch.cat([two_d_single_colour_grid + xy_offset for xy_offset in xy_offsets])

        return two_d_single_colour_grid, full_2d_grid

    def get_center_coords(self):

        two_d_single_colour_grid, full_2d_grid = self.get_full_2d_grid()

        colours_2d = torch.tensor([[i]*len(two_d_single_colour_grid) for i in range(7)]).flatten()

        z_offsets = self.get_all_z_offsets(colours_2d, self.hex_neighbors)

        shifted_2d_grid = torch.cat((full_2d_grid.to(self.device), z_offsets), dim = -1)

        full_3d_grid = torch.cat([shifted_2d_grid + torch.cat([torch.zeros(shifted_2d_grid.shape[0], 2).to(self.device) , i*self.z_dist*torch.ones(shifted_2d_grid.shape[0]).to(self.device)[:, None]], axis = -1) for i in range(-self.z_repeat, self.z_repeat+3)]) #TODO: That's a hack to make the box large enough

        colours_3d = torch.cat([colours_2d if i % 2 == 0 else colours_2d + 7 for i in range(2*(self.z_repeat + 1) +1)])

        # Construct the final tensors
        logit_tensor = F.one_hot(colours_3d, 14).float().to(self.device)

        self.center_coords = full_3d_grid
        self.center_logits = logit_tensor

    def get_all_z_offsets(self, colours_2d, neighbor_indices):
        """
        Calculate z-offsets using breadth-first filling from the first point.
        """
        # Keep the same offset calculations as before
        blue_purple = -self.trainable_offsets['purple_orange'] - self.trainable_offsets['orange_blue']
        pink_red = -self.trainable_offsets['orange_pink'] - self.trainable_offsets['red_orange']
        blue_red = -self.trainable_offsets['orange_blue'] - self.trainable_offsets['red_orange']
        blue_yellow = -self.trainable_offsets['pink_blue'] - self.trainable_offsets['yellow_pink']
        purple_red = -self.trainable_offsets['red_yellow'] - self.trainable_offsets['yellow_purple']
        red_green = -self.trainable_offsets['green_blue'] - blue_red
        yellow_green = -self.trainable_offsets['green_blue'] - blue_yellow
        green_purple = -red_green - purple_red
        purple_pink = -self.trainable_offsets['pink_blue'] - blue_purple
        pink_green = -green_purple - purple_pink
        green_orange = -self.trainable_offsets['orange_pink'] - pink_green
        orange_yellow = -yellow_green - green_orange

        non_trainable_offsets = {
            "blue_purple": blue_purple,
            "pink_red": pink_red,
            "blue_red": blue_red,
            "blue_yellow": blue_yellow,
            "purple_red": purple_red,
            "red_green": red_green,
            "yellow_green": yellow_green,
            "green_purple": green_purple,
            "purple_pink": purple_pink,
            "pink_green": pink_green,
            "green_orange": green_orange,
            "orange_yellow": orange_yellow
        }

        colour_dict = {
            0: 'blue',
            1: 'pink',
            2: 'yellow',
            3: 'green',
            4: 'red',
            5: 'orange',
            6: 'purple'
        }

        def get_offset(self, color1, color2):
            key = f"{color1}_{color2}"
            if key in self.trainable_offsets:
                return self.trainable_offsets[key]
            elif key in non_trainable_offsets:
                return non_trainable_offsets[key]
            key = f"{color2}_{color1}"
            if key in self.trainable_offsets:
                return -self.trainable_offsets[key]
            return -non_trainable_offsets[key]

        # Initialize z_offsets with infinity
        z_offsets = torch.full((len(colours_2d), 1), float('inf'), device=self.device)
        z_offsets[0] = 0  # Start point

        from collections import deque
        to_process = deque([0])  # Start with the first point
        processed = set()  # Keep track of points we've processed

        while to_process:
            current = to_process.popleft()
            if current in processed:
                continue
                
            current_color = colour_dict[colours_2d[current].item()]
            current_z = z_offsets[current].item()

            # Process each unprocessed neighbor
            for neighbor in neighbor_indices[current]:
                if neighbor not in processed:
                    neighbor_color = colour_dict[colours_2d[neighbor].item()]
                    offset = get_offset(self, current_color, neighbor_color)
                    z_offsets[neighbor] = current_z + offset
                    to_process.append(neighbor)

            processed.add(current)

        return z_offsets


    def forward(self, x):

        self.get_center_coords()

        if len(x.shape) == 3:
            reshape = True
            batch_size, n_circle_points, _ = x.shape
            x = x.flatten(0, 1)
        else:
            reshape = False

        # Voronoi diagram computation
        distances = torch.norm(x[:, None, :] - self.center_coords, dim=2)

        # shape of distances is (batch_size, num_centers) - for each point, the distance to all centers
        inverse_distances = 1 / distances

        # mean of center logits (or maybe center probabilities?!) weighted by inverse distances

        if self.temperature >= 0:
            if self.average_probabilities:
                center_probs = self.center_logits
                color_distribution = torch.softmax(inverse_distances * self.temperature, dim = 1) @ center_probs
            else:
                color_distribution = torch.softmax(inverse_distances @ self.center_logits * self.temperature, dim = 1)
        elif self.temperature == -1:
            # just take logits of the closest center
            closest_centers = torch.argmax(inverse_distances, dim=1)
            color_distribution = self.center_logits[closest_centers]
        else:
            raise ValueError(f"Invalid temperature {self.temperature}.") 

        if reshape:
            color_distribution = color_distribution.reshape(batch_size, n_circle_points, -1)

        return color_distribution
    
    def get_hex_neighbors(self, full_2d_grid, n_rows, n_cols):
        """
        Find the 6 neighbors for each point in a hexagonal grid.
        
        Args:
            full_2d_grid: Tensor of shape [N, 2] containing all points in the hexagonal grid
            n_rows: Number of rows in the base grid
            n_cols: Number of columns in the base grid
        """
        base_grid_size = (2 * n_rows + 1) * (2 * n_cols + 1)
        neighbor_indices = []
        
        # For each point
        for i in range(len(full_2d_grid)):
            neighbors = []
            point_coords = full_2d_grid[i]
            
            # Find all points that are exactly hex_size distance away
            for j in range(len(full_2d_grid)):
                if i != j:
                    dist = torch.norm(full_2d_grid[j] - point_coords)
                    # Use a small epsilon for floating point comparison
                    if abs(dist - self.hex_size) < 1e-5:
                        neighbors.append(j)
                        
            neighbor_indices.append(neighbors)
        
        return neighbor_indices