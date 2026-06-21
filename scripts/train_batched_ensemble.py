#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train many tiny MLP colorings in one vectorized process.")
    parser.add_argument("--ensemble-size", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=75000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-circle-points", type=int, default=8)
    parser.add_argument("--loss-log-every", type=int, default=1000)
    parser.add_argument("--output-root", type=Path, default=Path("/scratch/htc/npelleriti/agentic-almost-colorings/ensembles"))
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--base-seed", type=int, default=3000000)

    parser.add_argument("--learning-rates", default="0.001")
    parser.add_argument("--weight-decays", default="0.1")
    parser.add_argument("--good-coloring-weights", default="0.01")
    parser.add_argument("--temperatures", default="0.0")

    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--n-colours", type=int, default=6)
    parser.add_argument("--grid-size", type=float, default=6.0)
    parser.add_argument("--grid-input-scale", type=float, default=1.0)
    parser.add_argument("--p-norm", type=float, default=2.0)
    parser.add_argument("--parallelogram", default="2.0,1.0;1.0,2.0")
    parser.add_argument("--freeze-parallelogram", action="store_true")

    parser.add_argument("--n-hidden-layers", type=int, default=2)
    parser.add_argument("--n-hidden-units", type=int, default=64)
    parser.add_argument("--activation", choices=("sin", "relu", "tanh", "silu"), default="sin")
    parser.add_argument("--initialization", default="siren")
    parser.add_argument("--enable-residual-connections", action="store_true")
    return parser.parse_args()


def parse_parallelogram(raw: str, device: torch.device) -> torch.Tensor:
    rows = []
    for row in raw.split(";"):
        rows.append([float(item.strip()) for item in row.split(",") if item.strip()])
    tensor = torch.tensor(rows, dtype=torch.float32, device=device)
    if tensor.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 parallelogram, got shape {tuple(tensor.shape)}")
    return tensor


def activation_fn(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "sin":
        return torch.sin(x)
    if name == "relu":
        return F.relu(x)
    if name == "tanh":
        return torch.tanh(x)
    if name == "silu":
        return F.silu(x)
    raise NotImplementedError(name)


class BatchedResMLP(torch.nn.Module):
    def __init__(
        self,
        ensemble_size: int,
        input_dim: int,
        output_dim: int,
        n_hidden_layers: int,
        n_hidden_units: int,
        activation: str,
        initialization: str,
        disable_residual_connections: bool,
        parallelogram: torch.Tensor,
        trainable_parallelogram: bool,
        device: torch.device,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.activation = activation
        self.initialization = initialization
        self.disable_residual_connections = disable_residual_connections

        inv = torch.linalg.inv(parallelogram).expand(ensemble_size, -1, -1).clone()
        self.inv_transf_matrix = torch.nn.Parameter(inv, requires_grad=trainable_parallelogram)

        self.input_weight = torch.nn.Parameter(torch.empty(ensemble_size, n_hidden_units, input_dim, device=device))
        self.input_bias = torch.nn.Parameter(torch.empty(ensemble_size, n_hidden_units, device=device))
        self.hidden_weight = torch.nn.Parameter(
            torch.empty(ensemble_size, n_hidden_layers, n_hidden_units, n_hidden_units, device=device)
        )
        self.hidden_bias = torch.nn.Parameter(torch.empty(ensemble_size, n_hidden_layers, n_hidden_units, device=device))
        self.output_weight = torch.nn.Parameter(torch.empty(ensemble_size, output_dim, n_hidden_units, device=device))
        self.output_bias = torch.nn.Parameter(torch.empty(ensemble_size, output_dim, device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.initialization != "default" and "siren" not in self.initialization:
            raise ValueError("Initialization must be default or siren.")

        first_layer_scale = None
        if self.initialization != "default":
            parts = self.initialization.split("_")
            first_layer_scale = 30.0 if len(parts) == 1 else float(parts[1])

        self._init_linear(self.input_weight, self.input_bias, self.input_dim, first_layer_scale)
        for layer_idx in range(self.n_hidden_layers):
            self._init_linear(
                self.hidden_weight[:, layer_idx],
                self.hidden_bias[:, layer_idx],
                self.n_hidden_units,
                None,
            )
        self._init_linear(self.output_weight, self.output_bias, self.n_hidden_units, None)

    @staticmethod
    def _init_linear(
        weight: torch.Tensor,
        bias: torch.Tensor,
        fan_in: int,
        first_layer_scale: float | None,
    ) -> None:
        with torch.no_grad():
            std = 1.0 / fan_in if first_layer_scale is not None else math.sqrt(6.0 / fan_in)
            weight.uniform_(-std, std)
            bias.uniform_(-std, std)
            if first_layer_scale is not None:
                weight.mul_(first_layer_scale)

    @staticmethod
    def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("e...i,eoi->e...o", x, weight)
        bias_shape = (bias.shape[0],) + (1,) * (out.ndim - 2) + (bias.shape[1],)
        return out + bias.view(bias_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("e...d,edk->e...k", x, self.inv_transf_matrix)
        x = torch.remainder(x, 1.0)

        out = activation_fn(self.activation, self._linear(x, self.input_weight, self.input_bias))
        for layer_idx in range(self.n_hidden_layers):
            pre = self._linear(out, self.hidden_weight[:, layer_idx], self.hidden_bias[:, layer_idx])
            activated = activation_fn(self.activation, pre)
            out = activated if self.disable_residual_connections else out + activated
        return self._linear(out, self.output_weight, self.output_bias)

    def single_state_dict(self, index: int) -> dict[str, torch.Tensor]:
        state = {
            "input_layer.weight": self.input_weight[index].detach().cpu(),
            "input_layer.bias": self.input_bias[index].detach().cpu(),
            "output_layer.weight": self.output_weight[index].detach().cpu(),
            "output_layer.bias": self.output_bias[index].detach().cpu(),
        }
        for layer_idx in range(self.n_hidden_layers):
            state[f"hidden_layers.{layer_idx}.weight"] = self.hidden_weight[index, layer_idx].detach().cpu()
            state[f"hidden_layers.{layer_idx}.bias"] = self.hidden_bias[index, layer_idx].detach().cpu()
        return state


def sphere(size: tuple[int, ...], dim: int, p_norm: float, device: torch.device) -> torch.Tensor:
    if p_norm == 2:
        points = torch.randn(*size, dim, device=device)
    else:
        alpha = torch.full((*size, dim), 1.0 / p_norm, device=device)
        beta = torch.ones((), device=device)
        gamma = torch.distributions.Gamma(alpha, beta).sample()
        signs = 2 * torch.bernoulli(torch.full((*size, dim), 0.5, device=device)) - 1
        points = signs * gamma.pow(1.0 / p_norm)
    denom = torch.abs(points).pow(p_norm).sum(dim=-1, keepdim=True).pow(1.0 / p_norm)
    return points / denom.clamp_min(1e-12)


def compute_losses(
    anchor_logits: torch.Tensor,
    proximity_logits: torch.Tensor,
    temperatures: torch.Tensor,
    good_coloring_weights: torch.Tensor,
) -> torch.Tensor:
    anchor_probs = F.softmax(anchor_logits, dim=-1)
    proximity_probs = F.softmax(proximity_logits, dim=-1)

    same_colour_prob = (anchor_probs.unsqueeze(2) * proximity_probs)[..., :-1].sum(dim=-1)

    probs = torch.empty_like(same_colour_prob[..., 0])
    negative_temp = temperatures < 0
    zero_temp = temperatures == 0
    positive_temp = temperatures > 0

    if negative_temp.any():
        probs[negative_temp] = same_colour_prob[negative_temp].max(dim=-1).values
    if zero_temp.any():
        probs[zero_temp] = same_colour_prob[zero_temp].mean(dim=-1)
    if positive_temp.any():
        weights = torch.softmax(same_colour_prob[positive_temp] * temperatures[positive_temp, None, None], dim=-1)
        probs[positive_temp] = (same_colour_prob[positive_temp] * weights).sum(dim=-1)

    losses = -torch.log((1.0 - probs).clamp_min(1e-8))
    last_colour_prob = anchor_probs[..., -1] + proximity_probs[..., -1].mean(dim=2)
    losses = losses + good_coloring_weights[:, None] * last_colour_prob
    return losses.mean(dim=1)


def schedule_scale(step: int, n_steps: int) -> float:
    warmup = max(1, int(0.05 * n_steps))
    if step <= warmup:
        return step / warmup
    remaining = max(1, n_steps - warmup)
    return max(0.0, (n_steps - step) / remaining)


def adamw_step(
    params: list[torch.nn.Parameter],
    state: dict[int, tuple[torch.Tensor, torch.Tensor]],
    lrs: torch.Tensor,
    weight_decays: torch.Tensor,
    step: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    with torch.no_grad():
        for param in params:
            if param.grad is None:
                continue
            grad = param.grad
            if id(param) not in state:
                state[id(param)] = (torch.zeros_like(param), torch.zeros_like(param))
            exp_avg, exp_avg_sq = state[id(param)]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            view_shape = (lrs.shape[0],) + (1,) * (param.ndim - 1)
            lr = lrs.view(view_shape)
            wd = weight_decays.view(view_shape)
            param.mul_(1.0 - lr * wd)

            denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
            update = (exp_avg / bias_correction1) / denom
            param.add_(-lr * update)
            param.grad = None


def build_config(
    args: argparse.Namespace,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    good_coloring_weight: float,
    temperature: float,
    parallelogram: list[list[float]],
) -> dict:
    return {
        "run_id": 1,
        "seed": seed,
        "problem_name": "HadwigerNelson",
        "dim": args.dim,
        "n_colours": args.n_colours,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
        "training": {
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "grid_input_scale": args.grid_input_scale,
            "loss_fn": "log_prob",
            "grid_sizes": [args.grid_size] * args.dim,
            "p_norm": args.p_norm,
            "n_circle_points": args.n_circle_points,
            "temperature": temperature,
            "good_coloring": True,
            "good_coloring_weight": good_coloring_weight,
            "parallelogram": parallelogram,
            "trainable_parallelogram": 0 if args.freeze_parallelogram else 1,
        },
        "model": {
            "name": "ResMLP",
            "n_hidden_layers": args.n_hidden_layers,
            "n_hidden_units": args.n_hidden_units,
            "activation": args.activation,
            "initialization": args.initialization,
            "disable_residual_connections": not args.enable_residual_connections,
        },
        "metrics": {
            "plot_grid_size": 512,
            "val_grid_size": 128,
            "n_circle_points": 128,
            "log_metrics_every_k_steps": 1000,
            "log_imgs_every_k_steps": 1000,
            "log_model_every_k_steps": 100000,
            "enable_wandb_logging": False,
            "enable_eval": False,
            "enable_plots": False,
            "save_intermediate_models": False,
            "sync_models_to_wandb": False,
            "save_final_model": True,
            "loss_log_every_k_steps": args.loss_log_every,
        },
    }


def main() -> int:
    args = parse_args()
    if args.ensemble_size <= 0:
        raise ValueError("--ensemble-size must be positive")

    torch.manual_seed(args.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.base_seed)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    torch.backends.cudnn.benchmark = True

    sweep_id = args.sweep_id or f"ensemble_{int(time.time())}_{os.getpid()}"
    sweep_root = args.output_root / sweep_id
    sweep_root.mkdir(parents=True, exist_ok=True)

    learning_rate_values = parse_csv_floats(args.learning_rates)
    weight_decay_values = parse_csv_floats(args.weight_decays)
    good_weight_values = parse_csv_floats(args.good_coloring_weights)
    temperature_values = parse_csv_floats(args.temperatures)
    combos = list(itertools.product(learning_rate_values, weight_decay_values, good_weight_values, temperature_values))
    if not combos:
        raise ValueError("At least one hyperparameter combination is required.")

    seeds = torch.arange(args.base_seed, args.base_seed + args.ensemble_size, dtype=torch.long)
    combo_indices = [idx % len(combos) for idx in range(args.ensemble_size)]
    learning_rates = torch.tensor([combos[idx][0] for idx in combo_indices], dtype=torch.float32, device=device)
    weight_decays = torch.tensor([combos[idx][1] for idx in combo_indices], dtype=torch.float32, device=device)
    good_weights = torch.tensor([combos[idx][2] for idx in combo_indices], dtype=torch.float32, device=device)
    temperatures = torch.tensor([combos[idx][3] for idx in combo_indices], dtype=torch.float32, device=device)

    parallelogram = parse_parallelogram(args.parallelogram, device=device)
    model = BatchedResMLP(
        ensemble_size=args.ensemble_size,
        input_dim=args.dim,
        output_dim=args.n_colours,
        n_hidden_layers=args.n_hidden_layers,
        n_hidden_units=args.n_hidden_units,
        activation=args.activation,
        initialization=args.initialization,
        disable_residual_connections=not args.enable_residual_connections,
        parallelogram=parallelogram,
        trainable_parallelogram=not args.freeze_parallelogram,
        device=device,
    )

    params = [param for param in model.parameters() if param.requires_grad]
    opt_state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    bounds = torch.full((args.dim,), args.grid_size / 2.0, dtype=torch.float32, device=device)
    best_losses = torch.full((args.ensemble_size,), float("inf"), dtype=torch.float32, device=device)
    best_steps = torch.zeros((args.ensemble_size,), dtype=torch.long, device=device)
    final_losses = torch.zeros((args.ensemble_size,), dtype=torch.float32, device=device)
    best_state_dicts: list[dict[str, torch.Tensor] | None] = [None] * args.ensemble_size
    best_inv_transf_matrices: list[torch.Tensor | None] = [None] * args.ensemble_size

    run_dirs = []
    loss_files = []
    for idx in range(args.ensemble_size):
        combo_idx = combo_indices[idx]
        run_id = f"{sweep_id}_m{idx:03d}_seed{int(seeds[idx])}_combo{combo_idx:03d}"
        run_dir = sweep_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dirs.append(run_dir)
        loss_path = run_dir / "train_losses.csv"
        f = loss_path.open("w", encoding="utf-8", newline="")
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "learning_rate"])
        loss_files.append((f, writer))
        with (run_dir / "hparams.json").open("w", encoding="utf-8") as hpf:
            lr, wd, gw, temp = combos[combo_idx]
            json.dump(
                {
                    "model_index": idx,
                    "seed": int(seeds[idx]),
                    "combo_index": combo_idx,
                    "learning_rate": lr,
                    "weight_decay": wd,
                    "good_coloring_weight": gw,
                    "temperature": temp,
                },
                hpf,
                indent=2,
            )

    print(
        f"Training ensemble_size={args.ensemble_size} on {device} for {args.n_steps} steps; "
        f"{len(combos)} hyperparameter combos; output={sweep_root}",
        flush=True,
    )

    try:
        for step in tqdm(range(1, args.n_steps + 1)):
            anchor = (2.0 * bounds * torch.rand(args.ensemble_size, args.batch_size, args.dim, device=device)) - bounds
            unit = sphere((args.ensemble_size, args.n_circle_points), args.dim, args.p_norm, device)
            proximity = anchor[:, :, None, :] + unit[:, None, :, :]

            anchor_logits = model(anchor)
            proximity_logits = model(proximity)
            losses = compute_losses(anchor_logits, proximity_logits, temperatures, good_weights)
            losses.sum().backward()

            scale = schedule_scale(step, args.n_steps)
            adamw_step(params, opt_state, learning_rates * scale, weight_decays, step)

            if args.loss_log_every > 0 and (step % args.loss_log_every == 0 or step == args.n_steps):
                final_losses.copy_(losses.detach())
                improved = final_losses < best_losses
                best_losses[improved] = final_losses[improved]
                best_steps[improved] = step
                for idx in improved.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                    best_state_dicts[idx] = {
                        name: tensor.clone() for name, tensor in model.single_state_dict(idx).items()
                    }
                    best_inv_transf_matrices[idx] = model.inv_transf_matrix[idx].detach().cpu().clone()
                lr_now = (learning_rates * scale).detach().cpu().tolist()
                loss_now = final_losses.detach().cpu().tolist()
                for idx, (_, writer) in enumerate(loss_files):
                    writer.writerow([step, f"{loss_now[idx]:.10f}", f"{lr_now[idx]:.10g}"])
                min_loss = float(final_losses.min().detach().cpu())
                median_loss = float(final_losses.median().detach().cpu())
                print(f"step={step} min_loss={min_loss:.8f} median_loss={median_loss:.8f} lr_scale={scale:.6g}", flush=True)
    finally:
        for f, _ in loss_files:
            f.close()

    summary_rows = []
    for idx, run_dir in enumerate(run_dirs):
        state_dict = best_state_dicts[idx]
        inv_transf_matrix = best_inv_transf_matrices[idx]
        if state_dict is None or inv_transf_matrix is None:
            state_dict = model.single_state_dict(idx)
            inv_transf_matrix = model.inv_transf_matrix[idx].detach().cpu()
        torch.save(state_dict, run_dir / "trained_model.pt")
        lr, wd, gw, temp = combos[combo_indices[idx]]
        config = build_config(
            args=args,
            seed=int(seeds[idx]),
            learning_rate=lr,
            weight_decay=wd,
            good_coloring_weight=gw,
            temperature=temp,
            parallelogram=torch.linalg.inv(inv_transf_matrix).tolist(),
        )
        with (run_dir / "pipeline_config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        summary_rows.append(
            {
                "run_id": run_dir.name,
                "model_index": idx,
                "seed": int(seeds[idx]),
                "combo_index": combo_indices[idx],
                "learning_rate": lr,
                "weight_decay": wd,
                "good_coloring_weight": gw,
                "temperature": temp,
                "best_step": int(best_steps[idx].cpu()),
                "best_loss": float(best_losses[idx].cpu()),
                "final_step": args.n_steps,
                "final_loss": float(final_losses[idx].cpu()),
                "run_dir": str(run_dir),
            }
        )

    summary_rows.sort(key=lambda row: (row["best_loss"], row["run_id"]))
    summary_path = sweep_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved {args.ensemble_size} models to {sweep_root}")
    print(f"Best model: {summary_rows[0]['run_id']} best_loss={summary_rows[0]['best_loss']:.8f}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
