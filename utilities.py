# ===========================================================================
# Project:      Hadwiger-Nelson
# File:         utilities.py
# Description:  Utility functions
# ===========================================================================
from bisect import bisect_right
import os
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


def _discrete_cmap(name: str, n: int):
    try:
        return mpl.colormaps[name].resampled(n)
    except AttributeError:
        return plt.cm.get_cmap(name, n)
import numpy as np
import torch
import wandb
from tqdm.auto import tqdm
from torch.distributions.gamma import Gamma


class GeneralUtility:
    """Utility functions for general purposes."""

    @staticmethod
    def set_seed(seed: int):
        """
        Sets the seed for the current run.
        :param seed: seed to be used
        """
        if wandb.run is not None:
            wandb.config.update({'seed': seed})  # Push the seed to wandb

        # Set a unique random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(seed)  # This works if CUDA not available

    @staticmethod
    def save_model(model: torch.nn.Module, model_identifier: str, tmp_dir: str, sync: bool = False) -> str:
        """
        Saves the model's state_dict to a file.
        :param model_identifier: Name of the file type.
        :type model_identifier: str
        :param sync: Whether to sync the file to wandb.
        :type sync: bool
        :return: Path to the saved model.
        :rtype: str
        """
        fName = f"{model_identifier}_model.pt"
        fPath = os.path.join(tmp_dir, fName)

        # Remove all wrappers from the model
        while hasattr(model, 'base_model'):
            model = model.base_model

        # Only save models in their non-module version, to avoid problems when loading
        if hasattr(model, 'module'):
            model = model.module
        model_state_dict = model.state_dict()

        torch.save(model_state_dict, fPath)  # Save the state_dict

        if sync:
            wandb.save(fPath)
        return fPath

    @staticmethod
    def update_config_with_default(configDict, defaultDict):
        """Update config with default values recursively."""
        for key, default_value in defaultDict.items():
            if key not in configDict:
                configDict[key] = default_value
            elif isinstance(default_value, dict):
                configDict[key] = GeneralUtility.update_config_with_default(configDict.get(key, {}), default_value)
        return configDict

    @staticmethod
    def prepend_centering_scaling_to_module(model: torch.nn.Module, scaling: Optional[float],
                                            centering: Optional[float]) -> torch.nn.Module:
        """Prepends a scaling layer to the model, if scaling is not None."""

        class CenteringRescalingWrapper(torch.nn.Module):
            def __init__(self, base_model: torch.nn.Module, scale_value: float | None, centering_value: float | None):
                super().__init__()
                self.scale_factor = scale_value or 1.0
                self.centering = centering_value or 0.0
                self.base_model = base_model

            def forward(self, x):
                x = x - self.centering
                x = x * self.scale_factor
                x = self.base_model(x)
                return x

        CenteringRescalingWrapper.__name__ = model._get_name()
        return CenteringRescalingWrapper(base_model=model, scale_value=scaling, centering_value=centering)
    
    def prepend_parallelogram_transformation(model: torch.nn.Module,
                                             spanning_vectors: torch.Tensor,):
        
        transf_matrix = spanning_vectors
        inverse_transf_matrix = torch.linalg.inv(transf_matrix)
        
        class ParallelogramTransformationWrapper(torch.nn.Module):
            def __init__(self, base_model: torch.nn.Module, inv_transf_matrix: torch.Tensor):
                super().__init__()

                self.inv_transf_matrix = torch.nn.Parameter(inv_transf_matrix)
                self.inv_transf_matrix.requires_grad = False
                self.base_model = base_model

            def forward(self, x):
                x = x @ self.inv_transf_matrix
                x = x % 1.0
                x = self.base_model(x)
                return x
        ParallelogramTransformationWrapper.__name__ = model._get_name()
        return ParallelogramTransformationWrapper(base_model=model, inv_transf_matrix=inverse_transf_matrix)

    @staticmethod
    def verify_config(config: wandb.Config):  # ToDo: move this to problem classes
        """Verifies that the config is valid."""

        assert isinstance(config.training['grid_sizes'], list) or isinstance(config.training['grid_sizes'], tuple), f"Grid sizes must be an iterable, found type {type(config.training['grid_sizes'])}, concretely {config.training['grid_sizes']}"
        assert len(config.training['grid_sizes']) == config.dim, "Grid sizes must have the same length as the dimension."
        assert min(config.training['grid_sizes']) >= 1, "Grid sizes must be at least 1."

        assert isinstance(config.training['batch_size'], int), "Batch size must be an integer."

    @staticmethod
    def config_has_unknown_params(config: wandb.Config, defaults: dict) -> Tuple[bool, list]:
        """Recursively check if there are unknown config parameters, i.e. parameters that are not in the defaults."""
        does_not_exist = False
        params = []
        for config_key, config_val in config.items():
            if config_key not in defaults:
                does_not_exist = True
                params.append(config_key)
            elif isinstance(config_val, dict):
                if not does_not_exist:
                    does_not_exist, params = GeneralUtility.config_has_unknown_params(config_val, defaults[config_key])
        return does_not_exist, params

    @staticmethod
    def gnormal(*size, p=0.5, device=None, dtype=None, generator=None):
        alpha = torch.ones(*size, dtype=dtype, device=device)
        alpha *= torch.tensor(1 / p, dtype=dtype, device=device)
        beta = torch.tensor(1, dtype=dtype, device=device)
        gamma_distribution = Gamma(alpha, beta)
        r = 2 * torch.bernoulli(0.5 * torch.ones(*size, dtype=dtype, device=device), generator=generator) - 1
        return r * gamma_distribution.sample() ** (1 / p)

    @staticmethod
    def sphere(*size, d=2, p=2, device=None, dtype=None, generator=None):
        g = GeneralUtility.gnormal(*size, d, p=p, device=device, dtype=dtype, generator=generator)
        denom = (((torch.abs(g) ** p).sum(dim=-1)) ** (1 / p)).unsqueeze(-1)
        return g / denom

    @staticmethod
    @torch.no_grad()
    def get_parameter_count(model: torch.nn.Module) -> Tuple[int, int]:
        n_total = 0
        n_nonzero = 0
        param_list = ['weight', 'bias']
        for name, module in model.named_modules():
            for param_type in param_list:
                if hasattr(module, param_type) and not isinstance(getattr(module, param_type), type(None)):
                    p = getattr(module, param_type)
                    n_total += int(p.numel())
                    n_nonzero += int(torch.sum(p != 0))
        return n_total, n_nonzero

    @staticmethod
    def fill_dict_with_none(d):
        for key in d:
            if isinstance(d[key], dict):
                GeneralUtility.fill_dict_with_none(d[key])  # Recursive call for nested dictionaries
            else:
                d[key] = None
        return d

    @staticmethod
    @torch.no_grad()
    def get_parallelogram_coloring_metrics(
        model: torch.nn.Module,
        parallelogram: torch.Tensor,
        gridsize: int,
        n_circle_points: int,
        n_colours: int,
    ) -> dict[str, float]:
        device = parallelogram.device
        k, d = parallelogram.shape

        lin = torch.linspace(0.0, 1.0, gridsize, device=device)
        mesh = torch.meshgrid(*([lin] * k), indexing="ij")
        coeffs = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
        points = coeffs.matmul(parallelogram)
        grid_colours = model(points).argmax(dim=-1)

        bonus_index = n_colours - 1
        bonus_fraction = (grid_colours == bonus_index).float().mean().item()

        offsets = GeneralUtility.sphere(n_circle_points, d=d, p=2).to(device)
        conflicts = torch.zeros(points.size(0), device=device, dtype=torch.int32)
        for off in offsets:
            neigh_colours = model(points + off).argmax(dim=-1)
            conflicts += (grid_colours == neigh_colours)

        same_color_neighbor = conflicts > 0
        real_color_mask = grid_colours < bonus_index
        real_conflict_fraction = (real_color_mask & same_color_neighbor).float().mean().item()
        bad_fraction = bonus_fraction + real_conflict_fraction

        return {
            "bonus_fraction": bonus_fraction,
            "real_conflict_fraction": real_conflict_fraction,
            "bad_fraction": bad_fraction,
        }

    @staticmethod
    def get_parallelogram_eval(model:torch.nn.Module,
                               parallelogram:torch.Tensor,
                               gridsize:int,
                               n_circle_points:int,
                               n_colours:int,):
        metrics = GeneralUtility.get_parallelogram_coloring_metrics(
            model=model,
            parallelogram=parallelogram,
            gridsize=gridsize,
            n_circle_points=n_circle_points,
            n_colours=n_colours,
        )
        return metrics["bad_fraction"]
        

              

    
    @staticmethod
    def evaluate_on_grid(grid_bounds:tuple[float], 
                         model:torch.nn.Module, 
                         device:torch.device,
                         n_circle_points:int,
                         gridsize:int,
                         dim:int,
                         p_norm:int,
                         colour_distances:list[float],
                         verbose:bool = True,
                         good_coloring:bool = True,
                         concat_colours:bool = False,
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Evaluates the model on an equidistant grid of points in the domain
        for a given list of distances.

        :param grid_bounds: The bounds of the grid in each dimension.
        :param model: The model to evaluate.
        :param device: The device to evaluate the model on.
        :param n_circle_points: The number of circle points to sample per point.
        :param gridsize: The number of gridpoints in each dimension.
        :param dim: The dimension of the space (2 or 3).
        :param p_norm: The norm that induces the distance w.r.t. which unit-distance points are sampled.
        :param colour_distances: The distance to avoid for each colour.
        :param verbose: If True, show a progress bar over circle points.
        :param good_coloring: If True, zero out conflicts on points with the last colour.
        :param concat_colours: Deprecated; must be False.

        :return: Grid colours, conflicts per point, and per-point colour confidences.
        """
        assert not concat_colours, "concat_colours is no longer supported."

        x_array = torch.linspace(-grid_bounds[0], grid_bounds[0], gridsize)
        y_array = torch.linspace(-grid_bounds[1], grid_bounds[1], gridsize)

        if dim == 2:
            x, y = torch.meshgrid(x_array, y_array, indexing = "xy")
            flattened_input = torch.stack((x.flatten(), y.flatten()), dim=1)
            n_samples = gridsize**2
        
        elif dim == 3:
            z_array = torch.linspace(-grid_bounds[2], grid_bounds[2], gridsize)
            x, y, z = torch.meshgrid(x_array, y_array, z_array, indexing = "xy")
            flattened_input = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
            n_samples = gridsize**3
        else:
            raise NotImplementedError("Only 2D and 3D are supported.")

        repeated_distances = torch.tensor(colour_distances).expand(n_samples, -1)
        model_outs = model(flattened_input.to(device))
        grid_colours = model_outs.argmax(dim = -1)

        if torch.allclose(model_outs.sum(dim=-1), torch.ones(n_samples, device=device)):
            probs = model_outs
        else:
            probs = model_outs.softmax(dim = -1)
        grid_confidences = probs.max(dim = -1).values
        
        unit_circle_points = GeneralUtility.sphere(n_circle_points, d=dim, p=p_norm)
        repeated_circle_points = unit_circle_points[None, :, :].expand(n_samples, -1, -1)
        distances = repeated_distances[torch.arange(n_samples), grid_colours.to("cpu")]
        distance_circle_points = repeated_circle_points * distances[:, None, None]

        conflicts_per_point = torch.zeros(gridsize**dim, device=device)

        loop = tqdm(range(n_circle_points)) if verbose else range(n_circle_points)
        for i in loop:
            proximity_points = flattened_input + distance_circle_points[:, i, :]
            proximity_colours = model(proximity_points.to(device)).argmax(dim = -1)
            conflicts_per_point += (grid_colours == proximity_colours)

        # this is returned to calculate different metrics later
        if dim == 2:
            if good_coloring:
                grid_colours = grid_colours.reshape(gridsize, gridsize) 
                last_colour_mask = grid_colours == (model_outs.shape[1] - 1)
                conflicts_per_point = conflicts_per_point.reshape(gridsize, gridsize)
                conflicts_per_point[last_colour_mask] = 0
                return grid_colours, conflicts_per_point, grid_confidences.reshape(gridsize, gridsize)
            else:
                return grid_colours.reshape(gridsize, gridsize), conflicts_per_point.reshape(gridsize, gridsize), grid_confidences.reshape(gridsize, gridsize)
        elif dim == 3:
            if good_coloring:
                grid_colours = grid_colours.reshape(gridsize, gridsize, gridsize) 
                last_colour_mask = grid_colours == (model_outs.shape[1] - 1)
                conflicts_per_point = conflicts_per_point.reshape(gridsize, gridsize, gridsize)
                conflicts_per_point[last_colour_mask] = 0
                return grid_colours, conflicts_per_point, grid_confidences.reshape(gridsize, gridsize, gridsize)
            return grid_colours.reshape(gridsize, gridsize, gridsize), conflicts_per_point.reshape(gridsize, gridsize, gridsize), grid_confidences.reshape(gridsize, gridsize, gridsize)
        
    @staticmethod
    def evaluate_3D_grid(grid_bounds:tuple[float], 
                         model:torch.nn.Module, 
                         device:torch.device,
                         n_circle_points:int,
                         gridsize:int,
                         good_coloring:bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Essentially a copy of evaluate on grid, but for 3D. 
        Copy is necessary as 3D has some subtleties which would 
        make the already hard to read evaluate_on_grid function even worse.
        Note: Only supports OG Hadwiger Nelson Problem, not PolyChromaticNumber.
        """

        x_array = torch.linspace(-grid_bounds[0], grid_bounds[0], gridsize)
        y_array = torch.linspace(-grid_bounds[1], grid_bounds[1], gridsize)
        z_array = torch.linspace(-grid_bounds[2], grid_bounds[2], gridsize)

        x, y, z = torch.meshgrid(x_array, y_array, z_array, indexing = "xy")

        full_coords = torch.stack((x, y, z), dim=-1)

        grid_colours = torch.zeros(gridsize, gridsize, gridsize, device=device)
        for i in tqdm(range(gridsize)):
            grid_colours[i] = model(full_coords[i].to(device)).argmax(dim = -1)

        circle_points = GeneralUtility.sphere(n_circle_points, d=3, p=2)

        conflicts_per_point = torch.zeros(gridsize, gridsize, gridsize, device=device)

        for i in tqdm(range(n_circle_points)):

            proximity_points = full_coords + circle_points[i]
            proximity_colours = torch.zeros(gridsize, gridsize, gridsize, device=device)
            for j in range(gridsize):
                proximity_colours[j] = model(proximity_points[j].to(device)).argmax(dim = -1)
            conflicts_per_point += (grid_colours == proximity_colours)

        if good_coloring:
            # determine number of colors (dirty hack)
            test_tensor = torch.zeros(1, 3).to(device)
            n_colours = model(test_tensor).shape[1]
            last_colour_mask = grid_colours == (n_colours - 1)
            conflicts_per_point[last_colour_mask] = 0
        return grid_colours, conflicts_per_point
        

    
    @staticmethod
    def create_conflict_plot(grid_colours:torch.Tensor, 
                             conflicts_per_point:torch.Tensor,
                             grid_confidences:torch.Tensor,
                             grid_bounds:tuple[float],
                             gridsize:int,
                             n_colours:int,
                             save_path:str = None,
                             parallelogram:torch.Tensor = None,):
        
        conflicts_mask = conflicts_per_point > 0

        x_array = torch.linspace(-grid_bounds[0], grid_bounds[0], gridsize)
        y_array = torch.linspace(-grid_bounds[1], grid_bounds[1], gridsize)

        fig = plt.figure(figsize=(15, 5))

        plt.subplot(131)

        plt.title("Coloring")
        plt.pcolor(x_array, y_array, grid_colours.cpu(), cmap=_discrete_cmap("Pastel1", n_colours), vmin=0, vmax=n_colours - 1)
        ax = plt.gca()
        ax.set_aspect('equal')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6, width=1.5)
        ax.tick_params(axis='x', which='minor', length=3, width=1)
        ax.tick_params(axis='y', which='major', length=6, width=1.5)
        ax.tick_params(axis='y', which='minor', length=3, width=1)

        ax.set_xlim(-grid_bounds[0], grid_bounds[0])
        ax.set_ylim(-grid_bounds[1], grid_bounds[1])

        if parallelogram is not None:

            spanning_vectors = torch.tensor(parallelogram)

            corner_1 = spanning_vectors[0]
            corner_2 = spanning_vectors[1]
            corner_3 = spanning_vectors[0] + spanning_vectors[1]

            # Plot the parallelogram edges
            plt.plot([0, corner_1[0]], [0, corner_1[1]], color='black', linewidth=2)
            plt.plot([0, corner_2[0]], [0, corner_2[1]], color='black', linewidth=2)
            plt.plot([corner_1[0], corner_3[0]], [corner_1[1], corner_3[1]], color='black', linewidth=2)
            plt.plot([corner_2[0], corner_3[0]], [corner_2[1], corner_3[1]], color='black', linewidth=2)

        plt.subplot(132)

        num_conflicting_points = conflicts_mask.cpu().sum()
        plt.title(f"Points with conflicts (Ratio = {num_conflicting_points / gridsize ** 2:.06f})")
        plt.pcolor(x_array, y_array, conflicts_mask.cpu(), cmap=_discrete_cmap("binary", 2), vmin=0, vmax=1)
        #plt.colorbar(drawedges=True, location=cbar_location, shrink=shrink)
        ax = plt.gca()
        ax.set_aspect('equal')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='x', which='minor', length=3, width=1)  # Small ticks
        ax.tick_params(axis='y', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='y', which='minor', length=3, width=1)

        plt.subplot(133)

        plt.title("Confidence")
        plt.pcolor(x_array, y_array, grid_confidences.detach().cpu(), cmap=plt.cm.rainbow, vmin=0, vmax=1)
        ax = plt.gca()
        ax.set_aspect('equal')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='x', which='minor', length=3, width=1)  # Small ticks
        ax.tick_params(axis='y', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='y', which='minor', length=3, width=1)
        plt.colorbar(shrink=0.78)

        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def create_good_coloring_plot(grid_colours:torch.Tensor, 
                             conflicts_per_point:torch.Tensor,
                             grid_confidences:torch.Tensor,
                             grid_bounds:tuple[float],
                             gridsize:int,
                             n_colours:int,
                             save_path:str = None,
                             parallelogram:torch.Tensor = None,):
        
        conflicts_mask = conflicts_per_point > 0
        last_color_mask = grid_colours == (n_colours - 1)

        x_array = torch.linspace(-grid_bounds[0], grid_bounds[0], gridsize)
        y_array = torch.linspace(-grid_bounds[1], grid_bounds[1], gridsize)

        fig = plt.figure(figsize=(15, 5))

        plt.subplot(131)

        plt.title("Coloring")
        plt.pcolor(x_array, y_array, grid_colours.cpu(), cmap=_discrete_cmap("Pastel1", n_colours), vmin=0, vmax=n_colours - 1)
        ax = plt.gca()
        ax.set_aspect('equal')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6, width=1.5)
        ax.tick_params(axis='x', which='minor', length=3, width=1)
        ax.tick_params(axis='y', which='major', length=6, width=1.5)
        ax.tick_params(axis='y', which='minor', length=3, width=1)

        ax.set_xlim(-grid_bounds[0], grid_bounds[0])
        ax.set_ylim(-grid_bounds[1], grid_bounds[1])

        if parallelogram is not None:
            
            spanning_vectors = torch.tensor(parallelogram)

            a = spanning_vectors[0]
            b = spanning_vectors[1]
            center = 0.5 * (a + b)

            corner_0 = -center              # formerly at (0, 0)
            corner_1 = a - center
            corner_2 = b - center
            corner_3 = a + b - center

            # Plot the parallelogram edges
            plt.plot([corner_0[0], corner_1[0]], [corner_0[1], corner_1[1]], color='black', linewidth=0.5)
            plt.plot([corner_0[0], corner_2[0]], [corner_0[1], corner_2[1]], color='black', linewidth=0.5)
            plt.plot([corner_1[0], corner_3[0]], [corner_1[1], corner_3[1]], color='black', linewidth=0.5)
            plt.plot([corner_2[0], corner_3[0]], [corner_2[1], corner_3[1]], color='black', linewidth=0.5)

            # plot original parallelogram (to understand effects at the edges)
            plt.plot([0, a[0]], [0, a[1]], color='black', linewidth=0.5)
            plt.plot([0, b[0]], [0, b[1]], color='black', linewidth=0.5)
            plt.plot([a[0], a[0] + b[0]], [a[1], a[1] + b[1]], color='black', linewidth=0.5)
            plt.plot([b[0], a[0] + b[0]], [b[1], a[1] + b[1]], color='black', linewidth=0.5)

        plt.subplot(132)


        plot_mask = torch.zeros_like(conflicts_mask).float()
        plot_mask[conflicts_mask] = 1.
        plot_mask[last_color_mask] = 2.
        num_conflicting_points = conflicts_mask.cpu().sum()
        plt.title(f"Conflicts ratio = {num_conflicting_points / gridsize ** 2:.06f}, Last colour ratio: {last_color_mask.cpu().sum() / gridsize ** 2:.06f}")
        plt.pcolor(x_array, y_array, plot_mask.cpu(), cmap=_discrete_cmap("rainbow", 3), vmin=0, vmax=2)
        #plt.colorbar(drawedges=True, location=cbar_location, shrink=shrink)
        ax = plt.gca()
        ax.set_aspect('equal')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='x', which='minor', length=3, width=1)  # Small ticks
        ax.tick_params(axis='y', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='y', which='minor', length=3, width=1)

        plt.subplot(133)

        plt.title("Confidence")
        plt.pcolor(x_array, y_array, grid_confidences.detach().cpu(), cmap=plt.cm.rainbow, vmin=0, vmax=1)
        ax = plt.gca()
        ax.set_aspect('equal')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='x', which='minor', length=3, width=1)  # Small ticks
        ax.tick_params(axis='y', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='y', which='minor', length=3, width=1)
        plt.colorbar(shrink=0.78)

        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class SequentialSchedulers(torch.optim.lr_scheduler.SequentialLR):
    """
    Repairs SequentialLR to properly use the last learning rate of the previous scheduler when reaching milestones
    """

    def __init__(self, **kwargs):
        self.optimizer = kwargs['schedulers'][0].optimizer
        super(SequentialSchedulers, self).__init__(**kwargs)

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        self._schedulers[idx].step()


# Activation functions
class Sin(torch.nn.Module):
    """Sin activation function."""

    def forward(self, forward_input: torch.Tensor) -> torch.Tensor:
        return torch.sin(forward_input)
