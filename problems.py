# ===========================================================================
# Project:      Hadwiger-Nelson
# File:         problems.py
# Description:  Problems
# ===========================================================================
import abc
import os
from abc import ABC
import sys
from typing import Callable, Tuple, Optional

import torch
import wandb

import models as models
from utilities import GeneralUtility


class ProblemBaseClass(ABC):
    """Base class for Hadwiger-Nelson."""

    def __init__(self, config, device, debug, tmp_dir, **kwargs):
        self.config = config
        self.device = device
        self.debug = debug
        self.tmp_dir = tmp_dir

        self.n_colours = self.config['n_colours']
        self.dim = self.config['dim']
        self.tile_grid = self.config.training['tile_grid']
        self.p_norm = self.config.training['p_norm']
        self.grid_bounds = [0.5*b for b in self.config.training['grid_sizes']]
        self.grid_input_scale = self.config.training['grid_input_scale']

        self.network_input_dim = None

        self.loss_fn = self.get_loss_fn(
            loss_fn_name=self.config.training['loss_fn'],
            temperature=self.config.training['temperature'],
            good_coloring=self.config.training['good_coloring'],
            good_coloring_weight=self.config.training['good_coloring_weight'],
        )

    def compute_loss(self, model_outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        anchor_outputs, proximity_outputs = model_outputs['anchor_outputs'], model_outputs['proximity_outputs']
        return self.loss_fn(x=anchor_outputs, y=proximity_outputs)

    def set_model(self, reinit: bool, model_path: Optional[str] = None):
        sys.stdout.write(f"Loading model - reinit: {reinit} | path: {model_path if model_path else 'None specified'}.")
        if reinit:
            models_kwargs = {key: val for key, val in self.config.model.items()
                              if key != 'name' and val not in [None, 'None', 'none']}

            assert self.network_input_dim is not None, "Network input dimension not properly specified in inheriting class."

            model = getattr(models, self.config.model['name'])(
                input_dim=self.network_input_dim,
                output_dim=self.n_colours,
                num_colors=self.n_colours,
                grid_bound=self.grid_bounds[0],
                device=self.device,
                **models_kwargs,
            )

            model = GeneralUtility.prepend_centering_scaling_to_module(
                model=model, scaling=self.grid_input_scale, centering=0.,
            )
            if self.config["training"]["parallelogram"]:
                parallelogram = torch.tensor(self.config["training"]["parallelogram"]).to(self.device)
                model = GeneralUtility.prepend_parallelogram_transformation(
                    model=model, spanning_vectors=parallelogram,
                )
        else:
            model = self.model

        if model_path is not None:
            state_dict = torch.load(model_path, map_location=self.device)
            new_state_dict = {}
            for key, val in list(state_dict.items()):
                if key.startswith('base_model.'):
                    key = key.replace("base_model.", "")
                new_state_dict[key] = val
            model.load_state_dict(new_state_dict)

        model = model.to(device=self.device)
        self.model = model

    @staticmethod
    def get_loss_fn(
        loss_fn_name: str,
        temperature: Optional[float] = None,
        good_coloring: Optional[bool] = False,
        good_coloring_weight: float = 1.,
    ) -> Callable:
        """Defines the loss function. good_coloring adds a Lagrangian term for the last colour."""
        assert loss_fn_name in ['prob', 'log_prob'], f"Loss function {loss_fn_name} not implemented."

        softmax_fn = torch.nn.Softmax(dim=-1)

        def loss_fn(**kwargs):
            x, y = kwargs['x'], kwargs['y']
            x_probs = softmax_fn(x)
            y_probs = softmax_fn(y)
            prods_of_probabilities = x_probs * y_probs

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
                last_colour_prob = x_probs[:, :, -1].mean(dim=-1) + y_probs[:, :, -1].mean(dim=-1)
                loss += good_coloring_weight * last_colour_prob

            return loss

        return loss_fn

    def get_model_outputs(self, batch: dict) -> dict:
        anchor_logits = self.model(batch['anchor_points'])
        proximity_logits = self.model(batch['proximity_points'])
        n_circle_points = self.config.training["n_circle_points"]
        return {
            'anchor_outputs': anchor_logits[:, None, :].expand(-1, n_circle_points, -1),
            'proximity_outputs': proximity_logits,
        }

    def get_fraction_points_with_conflict(self) -> float:
        gridsize = self.config.metrics['val_grid_size']
        _, conflicts_per_point, _ = self.evaluate_on_grid(
            gridsize=gridsize,
            good_coloring=self.config["training"]["good_coloring"],
        )
        conflicts_mask = conflicts_per_point > 0
        return conflicts_mask.sum().item() / (gridsize**self.config["dim"])

    def get_last_color_ratio(self) -> float:
        arrays = [
            torch.linspace(-bound, bound, self.config.metrics['val_grid_size'])
            for bound in self.grid_bounds
        ]
        grids = torch.meshgrid(*arrays, indexing='ij')
        grid_input = torch.stack(grids, dim=-1).reshape(-1, len(self.grid_bounds))
        grid_colors = self.model(grid_input.to(self.device)).argmax(dim=-1)
        last_colour_ratio = (grid_colors == self.n_colours - 1).sum().item()
        last_colour_ratio /= self.config.metrics['val_grid_size']**len(self.grid_bounds)
        return last_colour_ratio

    @abc.abstractmethod
    def sample_points(self, **kwargs) -> dict[str, torch.Tensor]:
        pass

    @abc.abstractmethod
    def get_metrics(self, **kwargs) -> dict[str, float]:
        pass

    @abc.abstractmethod
    def evaluate_on_grid(self, gridsize: int, good_coloring: bool) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pass

    @abc.abstractmethod
    def log_plots(self, save_path, parallelogram=None) -> None:
        pass


class HadwigerNelson(ProblemBaseClass):
    """Vanilla Hadwiger-Nelson: unit-distance conflicts on sampled anchor/proximity pairs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.colour_distances = [1. for _ in range(self.n_colours)]
        self.network_input_dim = self.dim

    def sample_points(self, n_samples: int) -> dict[str, torch.Tensor]:
        anchor_points = (
            2 * torch.tensor(self.grid_bounds) * torch.rand((n_samples, self.dim))
            - torch.tensor(self.grid_bounds)
        )
        unit_circle_points = GeneralUtility.sphere(
            self.config.training["n_circle_points"], d=self.dim, p=self.p_norm,
        )
        proximity_points = anchor_points[:, None, :] + unit_circle_points

        if self.tile_grid:
            proximity_points = GeneralUtility.convert_to_tiling(proximity_points, self.grid_bounds)

        return {'anchor_points': anchor_points, 'proximity_points': proximity_points}

    def get_metrics(self, **kwargs) -> dict[str, float]:
        fraction_points_with_conflict = self.get_fraction_points_with_conflict()

        if self.config["training"]["good_coloring"]:
            last_color_ratio = self.get_last_color_ratio()
            return {
                "fraction_points_with_conflict": fraction_points_with_conflict,
                "last_color_ratio": last_color_ratio,
                "full_loss": last_color_ratio + fraction_points_with_conflict,
            }
        return {"fraction_points_with_conflict": fraction_points_with_conflict}

    def evaluate_on_grid(self, gridsize: int, good_coloring: bool) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.dim == 2:
            return GeneralUtility.evaluate_on_grid(
                model=self.model,
                grid_bounds=self.grid_bounds,
                gridsize=gridsize,
                device=self.device,
                colour_distances=self.colour_distances,
                n_circle_points=self.config.metrics['n_circle_points'],
                dim=self.dim,
                p_norm=self.p_norm,
                concat_colours=False,
                good_coloring=good_coloring,
                tile_grid=self.config["training"]["tile_grid"],
            )

        return *GeneralUtility.evaluate_3D_grid(
            model=self.model,
            grid_bounds=self.grid_bounds,
            gridsize=gridsize,
            device=self.device,
            n_circle_points=self.config.metrics['n_circle_points'],
            good_coloring=good_coloring,
        ), None

    def log_plots(self, save_path, parallelogram=None) -> None:
        save_path = os.path.join(save_path, "current_plot.png")
        grid_colours, conflicts_per_point, grid_confidences = self.evaluate_on_grid(
            gridsize=self.config.metrics['plot_grid_size'],
            good_coloring=self.config["training"]["good_coloring"],
        )

        if parallelogram is None and self.config["training"]["parallelogram"] is not None:
            parallelogram = self.config["training"]["parallelogram"]

        plot_kwargs = dict(
            grid_colours=grid_colours,
            conflicts_per_point=conflicts_per_point,
            grid_confidences=grid_confidences,
            grid_bounds=self.grid_bounds,
            gridsize=self.config.metrics['plot_grid_size'],
            n_colours=self.n_colours,
            save_path=save_path,
            parallelogram=parallelogram,
        )

        if self.config["training"]["good_coloring"]:
            GeneralUtility.create_good_coloring_plot(**plot_kwargs)
        else:
            GeneralUtility.create_conflict_plot(**plot_kwargs)

        wandb.log({"Colouring": wandb.Image(save_path)}, commit=False)

    def get_fine_eval(self, gridsize, n_circle_points):
        _, conflicts_per_point = GeneralUtility.evaluate_3D_grid(
            model=self.model,
            grid_bounds=self.grid_bounds,
            gridsize=gridsize,
            device=self.device,
            n_circle_points=n_circle_points,
        )
        conflicts_mask = conflicts_per_point > 0
        return conflicts_mask.sum().item() / (gridsize**self.config["dim"])
