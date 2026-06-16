## Improving almost colorings

This repository contains the codebase and tools to research and find **almost colorings** of the plane (and space) with $c$ colors. It is based on the paper: [https://arxiv.org/pdf/2501.18527](https://arxiv.org/pdf/2501.18527) which formalizes the mathematics and computational approach.

An almost-$c$-coloring of the plane is defined as a coloring of a subset of the plane with $c$ colors such that no two points in that subset at unit distance share the same color. In other words, we seek the minimum density of the "removed" (uncolored) set such that the remaining points can be colored with $c$ colors without monochromatic unit-distance pairs.

In our neural network formulation, we represent the "removed" set by introducing an additional $(c+1)$-th "bonus" (uncolored) color, and minimize its density using a Lagrangian relaxation approach.

## Algorithm 1: Automated almost-coloring formalization

In our prior work, we used the following algorithm:

1. **Initial training.** Train $p_\theta : \mathbb{R}^2 \to \Delta_{c+1}$ to minimize Equation (5) on a large enough box $[-R, R]^2$.
2. **Periodicity extraction.** Determine two vectors $v_1, v_2 \in \mathbb{R}^2$ with $0 \ll \angle(v_1, v_2) \ll \pi$ such that the coloring (largely) consists of tiling the parallelogram
$$\mathcal{P} = \{\alpha v_1 + \beta v_2 : \alpha, \beta \in [0, 1)\}$$
along the lattice $\Lambda = \{n_1 v_1 + n_2 v_2 : n_1, n_2 \in \mathbb{Z}\}$.
3. **Periodicity-constrained retraining.** Form the invertible change-of-basis matrix $M = [v_1 \ v_2] \in \mathbb{R}^{2 \times 2}$. Prepend the mapping $x \mapsto M^{-1}x \pmod 1$ to $p_\theta$, which enforces exact periodicity over $\Lambda$, and retrain.
4. **Discrete almost-coloring.** Discretize $\mathcal{P}$ into $kl$ copies of $\{\alpha v_1 / k + \beta v_2 / l : \alpha, \beta \in [0, 1)\}$ and determine a color for each parallelogram pixel by sampling $p_\theta$ at its respective center.
5. **Iteratively fix remaining conflicts.** Determine a discrete mask in which conflicts need to be avoided around each parallelogram pixel to obtain a formal coloring. Iteratively reduce any remaining conflicts by solving an auxiliary minimum edge cover problem and recoloring some parallelograms. After a fixed number of rounds, resolve any remaining conflicts by recoloring with the additional color $c + 1$ (the bonus color).

Here, we skip step 1 and 2 and directly start with step 3. The vectors themselves are implemented as trainable parameters as of now, so we can search/optimize different starting values. 

We want to focus our research mainly on **step 5**. After obtaining a discrete (constant on parallelogram pixels) coloring, it is not exactly clear how to optimize the pixel recoloring to minimize the density of the bonus color. Since the discretization introduces a finite "conflict mask" (specifying which pixel offsets on the torus can conflict at distance exactly 1), we can cast this as an Integer Linear Program (ILP) or Mixed Integer Linear Program (MILP), as prototyped in `scripts/verify_paralellogram_ip.py`. The goal is to find the assignment of real colors and the bonus color that minimizes the percentage of pixels assigned to the bonus color while guaranteeing that no two real-colored pixels at distance approximately 1 share the same color.

---

## Results Table

The values in this table correspond to known continuous analytical/geometric constructions (not to any specific discretization grid size). Finding a better value on **any** grid size is a success. Generally, the smaller the pixels (i.e., the larger the grid size), the better the discrete approximation can get.

| # colors | 1 | 2 | 3 | 4 | 5 | 6 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **best known** | 77.04% | 54.13% | 31.20% | 8.25% | 3.74% | 0.0149% |

---

## Getting Started & Offline Running

### Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. PyTorch is installed from PyTorch's CUDA 12.4 wheel index (`cu124`), matching drivers that report CUDA 12.4 in `nvidia-smi`.

**Prerequisites:** [uv](https://docs.astral.sh/uv/getting-started/installation/) and an NVIDIA GPU. Check your driver's max CUDA version with `nvidia-smi` (top-right "CUDA Version" line). If it differs from 12.4, edit the `pytorch-cu124` index in `pyproject.toml` (e.g. `cu126`, `cu128`) and re-run `uv sync --extra cuda`.

```bash
# One-time setup: create .venv and install dependencies (GPU PyTorch on Linux)
uv sync

# Verify GPU PyTorch
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If you need a different CUDA version, edit the `pytorch-cu124` index in `pyproject.toml` (e.g. `cu126`, `cu128`, `cu130`) and re-run `uv sync`. See [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for details.

For CPU-only development (no GPU), change the `pytorch-cu124` index in `pyproject.toml` to `pytorch-cpu` and re-run `uv sync`.

### Running Training
To start a training run:
```bash
uv run python main.py --debug
```
*Note: In debug mode (`--debug` in arguments), default parameters are used and a short runs/sweeps profile is executed. Without `--debug`, all config fields default to `None` for hyperparameter sweeps.*

### Local & Offline Runs
You do not need a connection to the Weights & Biases (W&B) cloud to run this code. You can disable W&B logging or run offline:
```bash
# Run completely offline (W&B metadata and runs are saved locally)
export WANDB_MODE=offline
python main.py --debug

# Disable W&B entirely
export WANDB_MODE=disabled
python main.py --debug
```

### Unified Local Pipeline (Train + ILP Verify)
Use the single pipeline command to train and then verify on a coarse grid (default `32x32`):
```bash
python run_pipeline.py
```

Useful flags:
```bash
# Keep W&B optional: disabled by default, enable only when needed
python run_pipeline.py --use-wandb

# Change training command or verification grid size
python run_pipeline.py --train-command "python main.py --debug" --eval-gridsize 32
```

### Persistent Local Checkpoints
When a training run completes, the trained model and plots are automatically copied from the temporary directory to the persistent path:
`./models/{run_id}/`
When using `run_pipeline.py`, a `pipeline_config.json` snapshot is also saved in the same run directory for local verification.
This ensures your results survive tempdir cleanup, making it easy to run evaluation scripts locally.

### Custom W&B Settings
If you do want to log to your own W&B account, set these environment variables before running:
```bash
export WANDB_PROJECT="your-project-name"
export WANDB_ENTITY="your-username"
python main.py --debug
```

---

## Ideas & Research Directions

* **Hyperparameter Tuning:** The hyperparameters in the training stage have not been thoroughly tuned. Small MLPs (2–4 hidden layers, 32–128 units) with `sin` activations and siren initialization perform well, but learning rates, schedules, and weight decay can be heavily optimized.
* **Bilevel Optimization for Lagrange Weight:** The formulation uses a Lagrangian term (`good_coloring_weight`) for the last color, which is painful to tune. Investigating a bilevel optimization strategy or a dynamic weight scheduler for the Lagrangian coefficient could make training more robust.
* **Alternating Parallelogram Optimization:** Making the parallelogram basis trainable alongside the neural network parameters is often unstable. Experiment with alternating optimization schemes: freeze the parallelogram basis and train the network, then freeze the network and train the basis (or use highly asymmetric learning rates).
* **Step 5 ILP Optimization (Connected-Component Decomposition):**
  * Writing the MILP is straightforward (see `scripts/verify_paralellogram_ip.py`), but for large/fine grids (e.g. 512x512 or larger), solving a single massive MILP is extremely slow or fails due to memory limits.
  * *Speedup Strategy:* Since conflicts are typically sparse and highly localized, we can build the conflict graph, extract its connected components, and solve a separate independent MILP for each component. Any non-conflicting components or pixels can remain frozen. This allows solving Step 5 on extremely fine discretizations (e.g., 1024x1024) in seconds!
* **Generalizing to 3D:** Once 2D coloring improves, generalize and apply the same pipeline to 3D (starting from `scripts/verify_parallelogram_3d_speedup.py`).

## Tasks

### Onboarding Task (Highly Recommended First Step)
* [x] **Build a unified local pipeline script** (`run_pipeline.py`) that handles the end-to-end training and evaluation loop. It:
  1. Trigger a training run (with custom configuration) and save the checkpoint.
  2. Parse the resulting run ID and configuration.
  3. Automatically run the discrete coloring MILP solver (`verify_paralellogram_ip.py`) on the trained checkpoint.
  4. Outputs the final, verified bonus color percentage.
  *This gives the agent a single, frictionless command to evaluate any code or hyperparameter changes instantly.*

### Core Research Tasks
* [ ] Find an almost 5-coloring with less than **3.74%** of the pixels covered by the bonus color (color index 5) on any discretization grid.
* [ ] Formalize the best constructions you can find for $c = 1, \dots, 6$. (Note that in the config one needs to pass $c+1$ total colors).
* [ ] Implement Connected-Component Decomposition in the ILP/MILP solver (`scripts/verify_paralellogram_ip.py`) to scale evaluations to grid sizes $\ge 512 \times 512$ efficiently.
* [ ] Design an alternating optimization schedule (joint/alternating coordinate descent) for training the network and the parallelogram basis vectors in `runner.py`.
* [ ] Experiment with dynamic/adaptive schedules for the Lagrange term `good_coloring_weight` to automate the tuning of the penalty weight.
* [ ] Generalize/apply findings to 3D. Only start this if you are done with 2D completely.
