# ===========================================================================
# Project:      Hadwiger-Nelson
# File:         problems.py
# Description:  Problems
# ===========================================================================
import abc
import os
from abc import ABC
import sys
from typing import Callable, Tuple, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import torch
import wandb
from tqdm import tqdm

import models as models
from utilities import GeneralUtility
from models import VoronoiCenters


class ProblemBaseClass(ABC):
    """Base class for algorithms, e.g. Hadwiger-Nelson."""

    def __init__(self, config, device, debug, tmp_dir, **kwargs):
        # Get run configuration
        self.config = config
        self.device = device
        self.debug = debug
        self.tmp_dir = tmp_dir

        # Useful variables
        self.n_colours = self.config['n_colours']  # Numer of colours in the problem
        self.dim = self.config['dim']  # Dimension of the problem
        self.tile_grid = self.config.training['tile_grid']
        self.p_norm = self.config.training['p_norm']
        self.grid_bounds = [0.5*b for b in self.config.training['grid_sizes']]
        self.grid_input_scale = self.config.training['grid_input_scale']

        # Variables to be set by inheriting classes
        self.network_input_dim = None   # The input dimension of the network

        self.loss_fn = self.get_loss_fn(loss_fn_name=self.config.training['loss_fn'], 
                                        temperature=self.config.training['temperature'],
                                        good_coloring=self.config.training['good_coloring'],
                                        good_coloring_weight=self.config.training['good_coloring_weight'])   # Define self.loss_fn

    def compute_loss(self, model_outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns the loss."""
        anchor_outputs, proximity_outputs = model_outputs['anchor_outputs'], model_outputs['proximity_outputs']

        if 'colours' in batch:
            loss = self.loss_fn(x=anchor_outputs, y=proximity_outputs, colours=batch['colours'])
        else:
            loss = self.loss_fn(x=anchor_outputs, y=proximity_outputs)
        return loss

    def set_model(self, reinit: bool, model_path: Optional[str] = None):
        """
        Returns the model.
        :param reinit: If True, the model is reinitialized.
        :type reinit: bool
        :param model_path: Path to the model.
        :type model_path: Optional[str]
        """
        sys.stdout.write(f"Loading model - reinit: {reinit} | path: {model_path if model_path else 'None specified'}.")
        if reinit:
            # Define the model
            models_kwargs = {key: val for key, val in self.config.model.items() if
                         key != 'name' and val not in [None, 'None', 'none']}
            
            assert self.network_input_dim is not None, "Network input dimension not properly specified in inheriting class."

            if self.config.model['name'] == 'DeepONet':
                trunk_input_dim = self.dim
                branch_input_dim = self.n_colours
            else:
                trunk_input_dim, branch_input_dim = None, None
                
            model = getattr(models, self.config.model['name'])(input_dim=self.network_input_dim,
                                                output_dim=self.n_colours,
                                                num_colors=self.n_colours,
                                                grid_bound=self.grid_bounds[0],
                                                device = self.device,
                                                trunk_input_dim=trunk_input_dim,
                                                branch_input_dim=branch_input_dim,
                                                **models_kwargs)
            
            model = GeneralUtility.prepend_centering_scaling_to_module(model=model, scaling=self.grid_input_scale, centering=0.)
            if self.config["training"]["parallelogram"]:
                parallelogram = torch.tensor(self.config["training"]["parallelogram"]).to(self.device)
                model = GeneralUtility.prepend_parallelogram_transformation(model=model, spanning_vectors=parallelogram)
        else:
            # The model has been initialized already
            model = self.model

        if model_path is not None:
            # Load the model
            state_dict = torch.load(model_path, map_location=self.device)

            # Remove the prefix "base_model." from the state_dict keys if existing
            new_state_dict = {}
            for key, val in list(state_dict.items()):
                # If there is at least one occurrence, remove the prefix
                if key.startswith('base_model.'):
                    # Remove all occurrences of "base_model."
                    key = key.replace("base_model.", "")
                new_state_dict[key] = val

            # Load the state_dict
            model.load_state_dict(new_state_dict)
        model = model.to(device=self.device)
        self.model = model

    @staticmethod
    def get_loss_fn(loss_fn_name: str, 
                    temperature: Optional[float] = None,
                    good_coloring: Optional[bool] = False,
                    good_coloring_weight: float = 1.) -> Callable:
        """
        Defines the loss function, both in the case of fixed distances and in the case of variable distances.
        good_coloring adds lagrangian term for last colour.
        """
        assert loss_fn_name in ['prob', 'log_prob', 'sqrt'], f"Loss function {loss_fn_name} not implemented."
        
        softmax_fn = torch.nn.Softmax(dim=-1)

        if loss_fn_name in ['prob', 'log_prob']:

            def loss_fn(**kwargs):

                x,y = kwargs['x'], kwargs['y']
                colours = kwargs['colours'] if 'colours' in kwargs else None

                if torch.allclose(x.sum(dim=-1), torch.tensor(1.)) and torch.allclose(y.sum(dim=-1), torch.tensor(1.)):
                    # hack for the case where the model outputs are already probabilities (voronoi)
                    prods_of_probabilities = x*y
                else:
                    x_probs = softmax_fn(x)
                    y_probs = softmax_fn(y)
                    prods_of_probabilities = x_probs * y_probs

                if colours is not None:
                    # Each batch of circle points belongs to a single colour/distance - select it
                    same_colour_prob = prods_of_probabilities[torch.arange(x.shape[0]), :, colours]
                else:
                    if good_coloring:
                        same_colour_prob = prods_of_probabilities[:, :, :-1].sum(dim=-1)
                    else:
                        same_colour_prob = prods_of_probabilities.sum(dim=-1)

                if temperature is not None and temperature > 0:
                    weights = torch.nn.Softmax(dim=-1)(same_colour_prob * temperature)
                    probs = (same_colour_prob * weights).sum(dim=1)
                else:
                    probs = same_colour_prob.max(dim=1)[0]

                loss = -torch.log(1. - probs) if loss_fn_name == 'log_prob' else probs

                if good_coloring:
                    # minimize occurence of last color
                    last_colour_prob = x_probs[:, :, -1].mean(dim=-1) + y_probs[:, :, -1].mean(dim=-1)
                    loss += good_coloring_weight * last_colour_prob
                
                return loss
            
        
        """
        elif loss_fn_name == 'sqrt':    # TODO: Requires revision
            def get_sqrt_loss(x, y, last_colour: bool):
                prods_of_probabilities = (softmax_fn(x) * softmax_fn(y))
                if last_colour:
                    return 1 - torch.sqrt(1 - prods_of_probabilities[:, -1])
                else:
                    sum_without_last_colour = (1. - torch.sqrt(1 - prods_of_probabilities[:, :-1])).sum(dim=1)
                    return sum_without_last_colour
                
            self.loss_fn = get_sqrt_loss
        """
        return loss_fn

    def get_model_outputs(self, batch: dict) -> dict:
        """Returns the model outputs and additional stuff."""

        if 'distances' in batch:
            anchor_inputs = torch.cat((batch['anchor_points'], batch['distances']), dim = 1)
            proximity_inputs = torch.cat((batch['proximity_points'], batch['distances'][:, None, :].expand(-1, self.config.training["n_circle_points"], -1)), dim = 2)
        else:
            anchor_inputs = batch['anchor_points']
            proximity_inputs = batch['proximity_points']
        anchor_logits = self.model(anchor_inputs)
        proximity_logits = self.model(proximity_inputs)
        return {'anchor_outputs': anchor_logits[:, None, :].expand(-1, self.config.training["n_circle_points"], -1), 'proximity_outputs': proximity_logits}

    def get_fraction_points_with_conflict(self, distances: Optional[list] = None) -> float:
        gridsize = self.config.metrics['val_grid_size']

        # The relevant information is in the conflicts_per_point tensor
        _, conflicts_per_point, _ = self.evaluate_on_grid(gridsize=gridsize,
                                                          colour_distances=distances,
                                                          good_coloring=self.config["training"]["good_coloring"])

        # for fraction_points_with_conflict we only need to know if there is any conflict
        conflicts_mask = conflicts_per_point > 0

        # here we don't care about the number of circle points anymore, just if any of those had a conflict
        fraction_points_with_conflict = (conflicts_mask).sum().item() / (gridsize**self.config["dim"])

        return fraction_points_with_conflict

    def get_last_color_ratio(self) -> float:

        arrays = [
        torch.linspace(-bound, bound, self.config.metrics['val_grid_size'])
        for bound in self.grid_bounds
    ]
    
        # Create n-dimensional grid
        grids = torch.meshgrid(*arrays, indexing='ij')
        
        # Stack the coordinates and reshape to (N, D) where D is the number of dimensions
        grid_input = torch.stack(grids, dim=-1).reshape(-1, len(self.grid_bounds))
        
        # Get model outputs
        model_outs = self.model(grid_input.to(self.device))
        grid_colors = model_outs.argmax(dim=-1)

        # Calculate ratio for the last color
        last_colour_mask = grid_colors == self.n_colours - 1
        last_colour_ratio = last_colour_mask.sum().item() / (self.config.metrics['val_grid_size']**len(self.grid_bounds))

        return last_colour_ratio

        # x_array = torch.linspace(-self.grid_bounds[0], self.grid_bounds[0], self.config.metrics['val_grid_size'])
        # y_array = torch.linspace(-self.grid_bounds[1], self.grid_bounds[1], self.config.metrics['val_grid_size'])

        # grid_input = torch.stack(torch.meshgrid(x_array, y_array), dim=-1).reshape(-1, 2)
        # model_outs = self.model(grid_input.to(self.device))
        # grid_colors = model_outs.argmax(dim=-1)

        # last_colour_mask = grid_colors == self.n_colours - 1
        # last_colour_ratio = last_colour_mask.sum().item() / (self.config.metrics['val_grid_size']**2)

        # return last_colour_ratio



    @abc.abstractmethod
    def sample_points(self, **kwargs) -> dict[str, torch.Tensor]:
        """Samples points from the problem domain."""
        pass

    @abc.abstractmethod
    def get_metrics(self, **kwargs) -> dict[str, float]:
        """
        :param model: Needed, since we need to evaluate the model to get the metrics.
        :param batch: A batch of points to evaluate the metrics on.
        """
        pass


class PolychromaticNumber(ProblemBaseClass):
    """Handles the case of PolychromaticNumber, where each colour has its own distance (d_1, ..., d_c) or interval of distances.
    In the latter case, the distance is sampled uniformly from the interval respective interval and _has_ to be passed to the NN.
    
    Special cases:
    - Hadwiger-Nelson: d_i = 1 for all colours
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_colour_distances([1.0] * self.n_colours)
        self.network_input_dim = self.dim + self.n_colours

    def _setup_colour_distances(self, colour_distances):
        if isinstance(colour_distances, (int, float)):
            colour_distances = [colour_distances for _ in range(self.n_colours)]
        colour_distances = [[c] if not isinstance(c, (list, tuple)) else c for c in colour_distances]
        colour_distances = [c if len(c) > 1 else [c[0], c[0]] for c in colour_distances]
        self.colour_distances = torch.tensor(colour_distances).T   # Shape: (2, n_colours)

    def sample_points(self, n_samples: Union[int, list]) -> dict[str, torch.Tensor]:
        """Sampling function given distance intervals."""
        n_circle_points = self.config["training"]["n_circle_points"]

        # Sample n_samples many points uniformly from the hypergrid
        # In dimension 2: [-x_bound, x_bound] x [-y_bound, y_bound]
        anchor_points = 2 * torch.tensor(self.grid_bounds)*torch.rand((n_samples, self.dim)) - torch.tensor(self.grid_bounds)

        if self.config.training['sample_all_colours']:
            # Sample n_samples many colours uniformly
            anchor_points = anchor_points.reshape(n_samples, 1, self.dim)
            anchor_points = anchor_points.expand(n_samples, self.n_colours, self.dim)
            anchor_points = anchor_points.reshape(n_samples * self.n_colours, self.dim)

            colours = torch.arange(self.n_colours)
            colours = colours.reshape(1, self.n_colours)
            colours = colours.expand(n_samples, self.n_colours)
            colours = colours.reshape(n_samples * self.n_colours)
            
        else:
            colours = torch.randint(0, self.n_colours, (n_samples,))

        # For each anchor point, sample n_colours many distances from the respective intervals in self.colour_distances
        # But only if the distance is not fixed, else sample once for the whole batch
        sampled_distances = (self.colour_distances[1] - self.colour_distances[0])*torch.rand((n_samples, self.n_colours)) + self.colour_distances[0]

        if self.config.training['sample_all_colours']:
            sampled_distances = sampled_distances.reshape(n_samples, 1, self.n_colours)
            sampled_distances = sampled_distances.expand(n_samples, self.n_colours, self.n_colours)
            sampled_distances = sampled_distances.reshape(n_samples * self.n_colours, self.n_colours)
        
        # Sample n_samples many points from the unit circle
        unit_circle_points = GeneralUtility.sphere(n_circle_points, d=self.dim, p=self.p_norm)
        

        # Multiply the unit circle points with the sampled distances corresponding to the right color in colours
        if self.config.training['sample_all_colours']:
            repeated_unit_circle_points = unit_circle_points[None, :, :].expand(n_samples*self.n_colours, -1, -1)
            distance_circle_points = repeated_unit_circle_points * sampled_distances[torch.arange(n_samples * self.n_colours), colours][:, None, None]
        
        else:
            repeated_unit_circle_points = unit_circle_points[None, :, :].expand(n_samples, -1, -1)
            distance_circle_points = repeated_unit_circle_points * sampled_distances[torch.arange(n_samples), colours][:, None, None]

        # Add the anchor points to the distance circle points
        proximity_points = anchor_points[:, None, :].expand(-1, self.config.training["n_circle_points"], -1) + distance_circle_points

        # Convert proximity_points to include tiling
        if self.tile_grid:
            proximity_points = GeneralUtility.convert_to_tiling(proximity_points, self.grid_bounds)
            
        return {'anchor_points': anchor_points, 'colours': colours, 'distances': sampled_distances,
                'proximity_points': proximity_points}  

    def get_metrics(self, **kwargs) -> dict[str, float]:  
        """
        Since we evaluate the metrics on a grid by default, we don't need a batch.
        :param distances: The distances to be used for the evaluation.
        """       
        conflict_fractions = [self.get_fraction_points_with_conflict(distances) for distances in kwargs['list_of_distances']]
        
        return {"min_fraction_points_with_conflict": min(conflict_fractions),
                "mean_fraction_points_with_conflict": sum(conflict_fractions) / len(conflict_fractions),
                "max_fraction_points_with_conflict": max(conflict_fractions)}

    def evaluate_on_grid(self, 
                         gridsize:int,
                         colour_distances:list[float],
                         good_coloring:bool) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Evaluates the model on an equidisant grid of points in the domain 
        for a given list of distances.

        :param gridsize: The number of points in each dimension of the grid.
        :param colour_distances: The distances to be used for the evaluation.

        :return: The colours of the grid points and the number of conflicts per point.
        """

        if good_coloring:

            print("You have specified good coloring, but this is not implemented for variable distances. Ignoring good coloring.")

        return GeneralUtility.evaluate_on_grid(model=self.model,
                                               grid_bounds=self.grid_bounds,
                                               gridsize=gridsize,
                                               device=self.device,
                                                colour_distances=colour_distances,
                                                n_circle_points=self.config.metrics['n_circle_points'],
                                                dim=self.dim,
                                                p_norm=self.p_norm,
                                                concat_colours=True,
                                                tile_grid=self.config["training"]["tile_grid"],
                                                good_coloring=False)
    
    def log_plots(self, save_path) -> None:
        for distances in self.config["metrics"]["eval_distances"]:
            save_path = os.path.join(self.tmp_dir, "current_plot.png")

            grid_colours, conflicts_per_point, grid_confidences = self.evaluate_on_grid(gridsize=self.config.metrics['plot_grid_size'],
                                                                      colour_distances=distances,
                                                                      good_coloring=False)
            

            GeneralUtility.create_conflict_plot(grid_colours=grid_colours,
                                                conflicts_per_point=conflicts_per_point,
                                                grid_confidences=grid_confidences,
                                                grid_bounds=self.grid_bounds,
                                                gridsize=self.config.metrics['plot_grid_size'],
                                                n_colours=self.n_colours,
                                                save_path=save_path,
                                                paralellogram=self.config["training"]["parallelogram"])
            
            wandb.log({f"Colouring" + str(distances): wandb.Image(save_path)}, commit=False)


                                               
class FixedDistances(PolychromaticNumber):

    """
    In the case where all distances are fixed, we can easily log metrics
    and plots.
    """
        # self.center_coords.requires_grad = False
        # self.center_logits.requires_grad = False

    def get_metrics(self, **kwargs) -> dict[str, float]:  
        """
        Since we evaluate the metrics on a grid by default, we don't need a batch.
        """
        if 'eval_distances' in kwargs and kwargs['eval_distances'] is not None:
            sys.stdout.write(f"Warning: We are evaluating on a grid, but the distances are fixed. Not using eval_distances.\n")

        fraction_points_with_conflict = self.get_fraction_points_with_conflict(distances=None)

        if self.config["training"]["good_coloring"]:
            last_color_ratio = self.get_last_color_ratio()
            return {"fraction_points_with_conflict": fraction_points_with_conflict, 
                    "last_color_ratio": last_color_ratio,
                    "full_loss": last_color_ratio + fraction_points_with_conflict}
        else:
            return {"fraction_points_with_conflict": self.get_fraction_points_with_conflict(distances=None)}
    
    def evaluate_on_grid(self, 
                         gridsize:int,
                         good_coloring: bool,
                         colour_distances: Optional[list] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Evaluates the model on an equidisant grid of points in the domain 
        for a given list of distances.

        :param gridsize: The number of points in each dimension of the grid.
        :param colour_distances: The distances to be used for the evaluation.

        :return: The colours of the grid points and the number of conflicts per point.
        """

        colour_distances = [float(distance) for distance in self.colour_distances[0]]    # Need to convert the [[a, a], [b, b]] list -> [a, b]

        return GeneralUtility.evaluate_on_grid(model=self.model,
                                               grid_bounds=self.grid_bounds,
                                               gridsize=gridsize,
                                               device=self.device,
                                                colour_distances=colour_distances,
                                                n_circle_points=self.config.metrics['n_circle_points'],
                                                dim=self.dim,
                                                p_norm=self.p_norm,
                                                concat_colours=True,
                                                tile_grid=self.config["training"]["tile_grid"],
                                                good_coloring=good_coloring)
    

    
    def log_plots(self, save_path, parallelogram=None) -> None:
        save_path = os.path.join(save_path, "current_plot.png")
        grid_colours, conflicts_per_point, grid_confidences = self.evaluate_on_grid(gridsize=self.config.metrics['plot_grid_size'],
                                                                                    good_coloring=self.config["training"]["good_coloring"])

        if self.config.model['name'] == 'VoronoiCenters' or self.config.model['name'] == 'ParametrizedVoronoiCenters':
            center_coords = self.model.base_model.center_coords.detach().cpu()
            center_logits = self.model.base_model.center_logits.detach().cpu()
        elif self.config.model['name'] == 'ParametrizedVoronoiCenters2D':
            self.model.base_model.get_center_coords_and_colours()
            center_coords = self.model.base_model.center_coords.detach().cpu()
            center_logits = self.model.base_model.center_logits.detach().cpu()
        else:
            center_coords = None
            center_logits = None
        
        if parallelogram is None:
            parallelogram = self.config["training"]["parallelogram"] if self.config["training"]["parallelogram"] is not None else None

        if self.config["training"]["good_coloring"]:
            GeneralUtility.create_good_coloring_plot(grid_colours=grid_colours,
                                            conflicts_per_point=conflicts_per_point,
                                            grid_confidences=grid_confidences,
                                            grid_bounds=self.grid_bounds,
                                            gridsize=self.config.metrics['plot_grid_size'],
                                            n_colours=self.n_colours,
                                            save_path=save_path,
                                            center_coords=center_coords,
                                            center_logits=center_logits,
                                            parallelogram=parallelogram)
        
        else:
            GeneralUtility.create_conflict_plot(grid_colours=grid_colours,
                                            conflicts_per_point=conflicts_per_point,
                                            grid_confidences=grid_confidences,
                                            grid_bounds=self.grid_bounds,
                                            gridsize=self.config.metrics['plot_grid_size'],
                                            n_colours=self.n_colours,
                                            save_path=save_path,
                                            center_coords=center_coords,
                                            center_logits=center_logits)
        
        wandb.log({f"Colouring": wandb.Image(save_path)}, commit=False)


class HadwigerNelson(FixedDistances):
    """Takes a point and samples the surrounding sphere."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.colour_distances = [1. for _ in range(self.n_colours)]
        self.network_input_dim = self.dim

    def sample_points(self, n_samples: int) -> dict[str, torch.Tensor]:
        """Sampling function."""

        anchor_points = 2 * torch.tensor(self.grid_bounds)*torch.rand((n_samples, self.dim)) - torch.tensor(self.grid_bounds)

        # Sample n_samples many points from the unit circle
        unit_circle_points = GeneralUtility.sphere(self.config.training["n_circle_points"], d=self.dim, p=self.p_norm)

        # Add the anchor points to the distance circle points
        proximity_points = anchor_points[:, None, :] + unit_circle_points

        # Convert proximity_points to include tiling
        if self.tile_grid:
            proximity_points = GeneralUtility.convert_to_tiling(proximity_points, self.grid_bounds)
            
        return {'anchor_points': anchor_points, 'proximity_points': proximity_points}
    
    def evaluate_on_grid(self, 
                         gridsize:int,
                         good_coloring: bool,
                         colour_distances: Optional[list] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Evaluates the model on an equidisant grid of points in the domain 
        for a given list of distances.

        :param gridsize: The number of points in each dimension of the grid.
        :param colour_distances: The distances to be used for the evaluation.

        :return: The colours of the grid points and the number of conflicts per point.
        """

        if self.dim == 2:

            return GeneralUtility.evaluate_on_grid(model=self.model,
                                                    grid_bounds=self.grid_bounds,
                                                    gridsize=gridsize,
                                                    device=self.device,
                                                    colour_distances=self.colour_distances,
                                                    n_circle_points=self.config.metrics['n_circle_points'],
                                                    dim=self.dim,
                                                    p_norm=self.p_norm,
                                                    concat_colours=False,
                                                    good_coloring=self.config.training['good_coloring'],
                                                    tile_grid=self.config["training"]["tile_grid"])

        elif self.dim == 3:

            return *GeneralUtility.evaluate_3D_grid(model=self.model,
                                                   grid_bounds=self.grid_bounds,
                                                    gridsize=gridsize,
                                                    device=self.device,
                                                    n_circle_points=self.config.metrics['n_circle_points'],
                                                    good_coloring=self.config["training"]["good_coloring"]), None

    def get_fine_eval(self, gridsize, n_circle_points):

        _, conflicts_per_point = GeneralUtility.evaluate_3D_grid(model=self.model,
                                                                    grid_bounds=self.grid_bounds,
                                                                    gridsize=gridsize,
                                                                    device=self.device,
                                                                    n_circle_points=n_circle_points)
        
        # for fraction_points_with_conflict we only need to know if there is any conflict
        conflicts_mask = conflicts_per_point > 0

        # here we don't care about the number of circle points anymore, just if any of those had a conflict
        fraction_points_with_conflict = (conflicts_mask).sum().item() / (gridsize**self.config["dim"])

        return fraction_points_with_conflict