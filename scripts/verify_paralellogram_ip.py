import argparse
from collections import defaultdict
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter
import torch
import pulp
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import networkx as nx
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from models import ResMLP
from utilities import GeneralUtility

# ----------------------
# Argument Parsing
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=str, required=True)
parser.add_argument('--eval_gridsize', type=int, default=512)
parser.add_argument('--wandb_project', type=str, default=os.getenv("WANDB_PROJECT", "2DHadwigerNelson"))
parser.add_argument('--wandb_entity', type=str, default=os.getenv("WANDB_ENTITY", "ais2t"))
parser.add_argument('--config_json', type=str, default=None,
                    help="Optional local config JSON. If provided, skips W&B config lookup.")
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help="Optional local checkpoint path. If provided, skips W&B checkpoint download.")
parser.add_argument('--output_dir', type=str, default='.',
                    help="Directory for fixed colorings, plots, and verification CSV output.")
parser.add_argument('--no_plot', action='store_true',
                    help="Skip final coloring plot generation.")
parser.add_argument('--solver_time_limit', type=int, default=360000,
                    help="CBC solver time limit in seconds.")
parser.add_argument('--solver_threads', type=int, default=1,
                    help="Number of CBC solver threads. Use 1 for deterministic single-threaded solves.")
parser.add_argument('--solver_backend', choices=('cbc', 'scip', 'cp_sat'), default='cbc',
                    help="Backend for component MILPs.")
parser.add_argument('--solver_gap_abs', type=float, default=None,
                    help="Absolute MIP optimality gap. For this problem this is bonus-color cells.")
parser.add_argument('--solver_gap_rel', type=float, default=None,
                    help="Relative MIP optimality gap.")
parser.add_argument('--solver_max_nodes', type=int, default=None,
                    help="Maximum branch-and-bound nodes for MIP solvers.")
parser.add_argument('--cbc_cuts', choices=('default', 'on', 'off'), default='default',
                    help="CBC cut generation setting.")
parser.add_argument('--cbc_presolve', choices=('default', 'on', 'off'), default='default',
                    help="CBC presolve setting.")
parser.add_argument('--cbc_strong', type=int, default=None,
                    help="CBC strong branching candidate count.")
parser.add_argument('--scip_path', type=str,
                    default=os.getenv("SCIP_PATH", "/software/opt-sw/scipoptsuite-10.0.2/bin/scip"),
                    help="Path to SCIP executable when --solver_backend=scip.")
parser.add_argument('--no_component_decomposition', action='store_true',
                    help="Use one global MILP over all active cells instead of connected components.")
parser.add_argument('--component_solver_msg', action='store_true',
                    help="Print CBC logs for each connected-component MILP.")
parser.add_argument('--neighbor_backend', choices=('fft', 'conv'), default='fft',
                    help="Backend for circular unit-distance mask queries.")
parser.add_argument('--mask_cache_dir', type=str, default=None,
                    help="Directory for cached base masks. Defaults to a mask_cache directory next to output_dir; set to 'none' to disable.")
parser.add_argument('--skip_vertex_cover', action='store_true',
                    help="Skip the approximate vertex-cover pre-repair and let the MILP handle all remaining conflicted cells.")
parser.add_argument('--active_edge_chunk_size', type=int, default=64,
                    help="Number of mask offsets to process at once when vectorizing active-active edge discovery.")
args = parser.parse_args()
output_dir = Path(args.output_dir)

# ----------------------
# Load run config + checkpoint
# ----------------------
run_config = None
checkpoint_path = args.checkpoint_path

if args.config_json is not None:
    with open(args.config_json, "r", encoding="utf-8") as f:
        run_config = json.load(f)

if run_config is None or checkpoint_path is None:
    api = wandb.Api()
    run = api.run(f"{args.wandb_entity}/{args.wandb_project}/{args.run_id}")
    run_config = run.config

    if checkpoint_path is None:
        for checkpoint_file in ('trained_model.pt', 'step_32768_model.pt', 'step_65536_model.pt'):
            try:
                ckpt = run.file(checkpoint_file).download(root=f"./models/{args.run_id}", replace=True)
                checkpoint_path = ckpt.name
                break
            except Exception:
                continue

if checkpoint_path is None:
    local_candidates = [
        f"./models/{args.run_id}/trained_model.pt",
        f"./models/{args.run_id}/step_32768_model.pt",
        f"./models/{args.run_id}/step_65536_model.pt",
    ]
    checkpoint_path = next((p for p in local_candidates if os.path.exists(p)), None)

if checkpoint_path is None:
    raise FileNotFoundError("Could not resolve checkpoint path from arguments, W&B, or local models directory.")


# ----------------------
# for plotting
# ----------------------
def get_cmap():

    colors = [
      '#FFD6A5',  # light orange"
      '#FDFFB6',  # light yellow
      '#CAFFBF',  # light green
      '#9BF6FF',  # light turquoise
      '#A0C4FF',  # light blue

      '#FFADAD',  # light red
   ]
    
    pastel_cmap = ListedColormap(colors, name='pastel')

    return pastel_cmap

def plot_parallelogram_coloring(starts, values, v1, v2, N, filename):

    dv1 = v1.cpu() / N
    dv2 = v2.cpu() / N
    starts = starts.reshape(N*N, 2).cpu()
    values = values.flatten()
    
    cmap = get_cmap()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for p, val in tqdm(zip(starts, values)):
        corners = np.stack([
            p,
            p + dv1,
            p + dv1 + dv2,
            p + dv2
        ])
        polygon = Polygon(corners, closed=True, facecolor=cmap(val / 5), edgecolor='none')
        ax.add_patch(polygon)
    
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.axis('off')
    plt.savefig(filename)
    plt.close()

# ----------------------
# Load Model
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim = run_config["dim"]

model = ResMLP(input_dim=input_dim, output_dim=run_config['n_colours'], device=device, **run_config["model"])
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device).eval()

parallelogram = torch.tensor(run_config["training"]["parallelogram"], device=device)
model = GeneralUtility.prepend_parallelogram_transformation(model, spanning_vectors=parallelogram)

# ----------------------
# Evaluate Model
# ----------------------
gridsize = args.eval_gridsize
n_colours = run_config['n_colours']


# ----------------------
# Compute Base Mask (once)
# ----------------------
def compute_base_mask(parallelogram, N, radius=1.0):
    v1, v2 = parallelogram[0], parallelogram[1]
    lin = torch.linspace(0, 1, N+1)[:-1]
    a, b = torch.meshgrid(lin, lin, indexing='ij')
    starts = (a.reshape(-1, 1) * v1 + b.reshape(-1, 1) * v2).cpu()
    dv1, dv2 = v1 / N, v2 / N
    base_mask = []
    tile_offsets = torch.tensor([[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1]], dtype=torch.float64)

    for offset in tile_offsets:
        shift = offset[0]*v1 + offset[1]*v2
        shifted_starts = starts + shift
        for idx, p in tqdm(enumerate(shifted_starts)):
            corners = torch.stack([p, p + dv1, p + dv1 + dv2, p + dv2])
            for center in [torch.tensor([0.0, 0.0]), dv1, dv1+dv2, dv2]:
                dists_sq = ((corners - center)**2).sum(dim=1)
                inside = dists_sq <= radius**2
                if not torch.all(inside) and torch.any(inside):
                    i, j = idx // N, idx % N
                    base_mask.append((i - 0, j - 0))  # relative to origin
                    break
    return starts, base_mask

def _resolve_mask_cache_dir():
    if args.mask_cache_dir == "none":
        return None
    if args.mask_cache_dir is not None:
        return Path(args.mask_cache_dir)
    return output_dir.parent / "mask_cache"


def _mask_cache_path(parallelogram, N, radius):
    cache_dir = _resolve_mask_cache_dir()
    if cache_dir is None:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    key_data = [
        np.asarray(parallelogram.detach().cpu().numpy(), dtype=np.float32).tobytes(),
        str(N).encode("ascii"),
        repr(float(radius)).encode("ascii"),
    ]
    digest = hashlib.sha256(b"|".join(key_data)).hexdigest()[:20]
    return cache_dir / f"base_mask_{N}_{digest}.npy"


def compute_base_mask_vec(parallelogram, N, radius=1.0, device='cpu'):
    v1, v2 = parallelogram[0], parallelogram[1]
    lin = torch.linspace(0, 1, N + 1, device=device, dtype=v1.dtype)[:-1]
    a, b = torch.meshgrid(lin, lin, indexing='ij')
    starts_grid = a.unsqueeze(-1) * v1 + b.unsqueeze(-1) * v2  # [N, N, 2]

    cache_path = _mask_cache_path(parallelogram, N, radius)
    if cache_path is not None and cache_path.exists():
        base_mask = np.load(cache_path).astype(np.int64, copy=False).tolist()
        print(f"Loaded cached base mask from {cache_path}")
        return starts_grid.reshape(-1, 2).cpu(), base_mask

    dv1, dv2 = v1/N, v2/N
    corner_offs = torch.stack([
        torch.zeros(2, device=device),
        dv1,
        dv1 + dv2,
        dv2
    ], dim=0)  # [4,2]
    tile_offsets = torch.tensor(
        [[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1]],
        device=device,
        dtype=v1.dtype,
    )

    base_mask = []
    radius_sq = radius * radius

    for tile_offset in tile_offsets:
        tile_partial = torch.zeros((N, N), dtype=torch.bool, device=device)
        shift = tile_offset[0] * v1 + tile_offset[1] * v2
        corners = starts_grid.unsqueeze(0) + shift + corner_offs[:, None, None, :]
        for center in corner_offs:
            d2 = ((corners - center.view(1, 1, 1, 2)) ** 2).sum(dim=-1)
            inside = d2 <= radius_sq
            tile_partial |= inside.any(dim=0) & (~inside.all(dim=0))

        base_mask.extend(torch.nonzero(tile_partial, as_tuple=False).tolist())

    if cache_path is not None:
        np.save(cache_path, np.asarray(base_mask, dtype=np.int32))
        print(f"Saved base mask cache to {cache_path}")

    return starts_grid.reshape(-1, 2).cpu(), base_mask

#starts, base_mask = compute_base_mask(parallelogram.cpu(), gridsize)
mask_start = perf_counter()
starts, base_mask = compute_base_mask_vec(parallelogram, gridsize, device=device)
print(f"Prepared base mask and grid starts in {perf_counter() - mask_start:.2f}s", flush=True)

starts_reshaped = starts.reshape(gridsize, gridsize, -1)
model_eval_start = perf_counter()
with torch.no_grad():
    outputs = model(starts.to(device)).argmax(dim=1).cpu()
    grid_coloring = outputs.reshape(gridsize, gridsize).numpy()
print(f"Evaluated model on verifier grid in {perf_counter() - model_eval_start:.2f}s", flush=True)


# ----------------------
# Helper functions
# ----------------------
def one_hot_encode(grid: torch.Tensor, C: int) -> torch.Tensor:
    """
    Turn [H,W] int labels 0..C-1 into [C,H,W] float one‑hot.
    """
    return F.one_hot(grid.long(), C).permute(2, 0, 1).float()

def build_mask_kernel(mask_offsets: np.ndarray, device: str):
    """
    Build a single conv‑kernel [1,1,Kh,Kw] with 1’s at each (di,dj) in mask_offsets.
    Also returns padding tuple for circular pad.
    """
    dis, djs = mask_offsets[:,0], mask_offsets[:,1]
    di_min, di_max = int(dis.min()), int(dis.max())
    dj_min, dj_max = int(djs.min()), int(djs.max())
    Kh, Kw = di_max - di_min + 1, dj_max - dj_min + 1

    kernel = torch.zeros((1,1,Kh,Kw), device=device)
    for di, dj in mask_offsets:
        ki, kj = int(di - di_min), int(dj - dj_min)
        kernel[0,0,ki,kj] = 1.0

    # padding = (left, right, top, bottom)
    pad = (-dj_min, dj_max, -di_min, di_max)
    return kernel, pad

def build_fft_mask_kernel(mask_offsets: np.ndarray, H: int, W: int, device: str):
    """
    Build the Fourier transform of the circular neighbor mask for FFT-based
    torus convolution. A neighbor offset (di,dj) contributes x[i+di,j+dj],
    so the convolution kernel is placed at (-di,-dj).
    """
    kernel = torch.zeros((H, W), device=device)
    for di, dj in mask_offsets:
        kernel[(-int(di)) % H, (-int(dj)) % W] = 1.0
    return torch.fft.rfft2(kernel)

def count_same_color_neighbors(onehot: torch.Tensor,
                               mask_kernel: torch.Tensor,
                               pad: tuple) -> torch.Tensor:
    """
    onehot: [C,H,W] float
    mask_kernel: [1,1,Kh,Kw] for conv, [H,W//2+1] Fourier kernel for fft
    pad: (left, right, top, bottom) for conv, None for fft
    Returns: [C,H,W] int counts of same-color neighbors.
    """
    if pad is None:
        H, W = onehot.shape[-2:]
        counts = torch.fft.irfft2(
            torch.fft.rfft2(onehot) * mask_kernel,
            s=(H, W),
        )
        return counts.round_().clamp_min_(0.0)

    # pad so wrap-around over the torus
    x = F.pad(onehot.unsqueeze(0), pad=pad, mode='circular')  # [1,C,H',W']
    # build grouped weight for C channels
    C = onehot.shape[0]
    weight = mask_kernel.repeat(C, 1, 1, 1)                   # [C,1,Kh,Kw]
    counts = F.conv2d(x, weight=weight, groups=C)             # [1,C,H,W]
    return counts.squeeze(0)                                  # -> [C,H,W]


def compute_used_colors(onehot: torch.Tensor,
                        mask_kernel: torch.Tensor,
                        pad: tuple) -> torch.Tensor:
    """
    onehot: [C,H,W] float
    mask_kernel: [1,1,Kh,Kw] for conv, [H,W//2+1] Fourier kernel for fft
    pad: (left, right, top, bottom) for conv, None for fft
    Returns: [C,H,W] bool indicating if any neighbor has that color.
    """
    return count_same_color_neighbors(onehot, mask_kernel, pad) > 0


def graph_min_vertex_cover(conflict_pairs):
    """
    Given list of edges (u,v) in pixel‑coordinate space, build graph and return approx vertex cover.
    """
    G = nx.Graph()
    G.add_edges_from(conflict_pairs)
    return nx.algorithms.approximation.min_weighted_vertex_cover(G)


def build_conflict_pairs_sparse(grid, conflict_mask, mask_offsets, chunk_size=128):
    """
    Build same-color conflict edges by looking only from currently-conflicted
    cells instead of rolling the whole grid once per mask offset.
    """
    conflict_idxs = torch.nonzero(conflict_mask, as_tuple=False)
    if conflict_idxs.numel() == 0:
        return []

    H, W = grid.shape
    ci = conflict_idxs[:, 0]
    cj = conflict_idxs[:, 1]
    conflict_colours = grid[ci, cj]
    offsets = torch.as_tensor(mask_offsets, device=grid.device, dtype=torch.long)

    conflict_pairs = []
    for chunk_start in range(0, offsets.shape[0], chunk_size):
        offset_chunk = offsets[chunk_start:chunk_start + chunk_size]
        ni = (ci.unsqueeze(0) + offset_chunk[:, 0:1]) % H
        nj = (cj.unsqueeze(0) + offset_chunk[:, 1:2]) % W
        matches = grid[ni, nj] == conflict_colours.unsqueeze(0)
        hits = torch.nonzero(matches, as_tuple=False)
        if hits.numel() == 0:
            continue

        hit_offsets = hits[:, 0]
        hit_cells = hits[:, 1]
        left_i = ci[hit_cells]
        left_j = cj[hit_cells]
        hit_di = offset_chunk[hit_offsets, 0]
        hit_dj = offset_chunk[hit_offsets, 1]
        right_i = (left_i + hit_di) % H
        right_j = (left_j + hit_dj) % W

        left_nodes = torch.stack((left_i, left_j), dim=1).cpu().tolist()
        right_nodes = torch.stack((right_i, right_j), dim=1).cpu().tolist()
        conflict_pairs.extend(
            (tuple(left), tuple(right))
            for left, right in zip(left_nodes, right_nodes)
        )

    return conflict_pairs

# ----------------------
# Build static data
# ----------------------
# base_mask is a list of (i,j) offsets computed alongside the model grid above.
mask_offsets = np.array(base_mask, dtype=int)

print(f"Computed base mask. {len(base_mask)=}, {base_mask[0]=}, mask_offsets={mask_offsets.shape=}")

if args.neighbor_backend == "fft":
    kernel_start = perf_counter()
    mask_kernel = build_fft_mask_kernel(mask_offsets, gridsize, gridsize, device)
    pad = None
    print(f"Computed FFT mask kernel. {mask_kernel.shape=} in {perf_counter() - kernel_start:.2f}s", flush=True)
else:
    kernel_start = perf_counter()
    mask_kernel, pad = build_mask_kernel(mask_offsets, device)
    print(f"Computed mask kernel. {mask_kernel.shape=}, pad={pad} in {perf_counter() - kernel_start:.2f}s", flush=True)

# ----------------------
# Convert initial coloring to torch
# ----------------------
grid = torch.from_numpy(grid_coloring).long().to(device)  # [H,W]
H, W = grid.shape
C = int(grid.max()) + 1

onehot = one_hot_encode(grid, C)  # [C,H,W]

print(f"{onehot.shape=}, {grid.shape=}, {C=}")

# # ----------------------
# # Iterative fix loop
# # ----------------------
# max_iters = 10
# for iteration in tqdm(range(max_iters)):
#     # 1) count same-color neighbors → [C,H,W]
#     neighbor_counts = count_same_color_neighbors(onehot, mask_kernel, pad)

#     # 2) detect conflicts: [H,W] bool
#     # Flatten grid and neighbor_counts to pick each pixel’s own‐color count:
#     flat_colors = grid.view(-1).long()             # [H*W]
#     flat_counts = neighbor_counts.view(C, -1)      # [C, H*W]
#     flat_idx    = torch.arange(H*W, device=grid.device)

#     # select count for each pixel’s color, then reshape back to [H,W]
#     selected     = flat_counts[flat_colors, flat_idx]   # [H*W]
#     conflict_mask = (selected.view(H, W) > 0)          # [H,W]

#     # 3) compute used-colors mask: [C,H,W]
#     used = compute_used_colors(onehot, mask_kernel, pad)

#     # 4) five_mask, five_fixable_mask, conflict_fixable_mask
#     five_mask = (grid == (C-1))                       # color C-1 is “5”
#     five_fixable = five_mask & (~used.any(dim=0))
#     conflict_fixable = conflict_mask & (~used.any(dim=0))

#     # 5) Gather conflict pixel coords:
#     idxs = torch.nonzero(conflict_mask, as_tuple=False)  # expect [N,2]
#     assert idxs.dim() == 2 and idxs.size(1) == 2, f"got idxs shape {idxs.shape}"

#     # conflict_pairs = []
#     # for i, j in tqdm(idxs.tolist()):
#     #     cc = int(grid[i, j])
#     #     for di, dj in mask_offsets:
#     #         ni, nj = (i + di) % H, (j + dj) % W
#     #         if int(grid[ni, nj]) == cc:
#     #             # add the tuple of tuples
#     #             conflict_pairs.append(((i, j), (ni, nj)))

#     conflict_pairs = []
#     cm = conflict_mask  # [H,W] bool

#     for di, dj in mask_offsets:
#         # 1) roll the grid by (-di,-dj) so that neighbor at (i+di,j+dj) moves to (i,j)
#         rolled = torch.roll(grid, shifts=(-di, -dj), dims=(0, 1))  # [H,W]

#         # 2) find where both a conflict *at* (i,j) and the neighbor *agrees* in color:
#         match = cm & (rolled == grid)  # [H,W] bool

#         if not match.any():
#             continue

#         # 3) get their coords
#         idxs = torch.nonzero(match, as_tuple=False)  # [K,2]

#         # 4) build pairs
#         #    original pixel (i,j) and its neighbor (i+di, j+dj)
#         for i, j in idxs.tolist():
#             ni = (i + di) % H
#             nj = (j + dj) % W
#             conflict_pairs.append(((i, j), (ni, nj)))


#     # 6) compute vertex-cover
#     min_vc = graph_min_vertex_cover(conflict_pairs)

#     # 7) apply fixes
#     #    start from scratch each iteration
#     new_grid = grid.clone()

#     # fix all five_fixable → assign lowest unused color
#     for c in range(C-1):
#         mask_c = five_fixable & (~used[c])
#         new_grid[mask_c] = c

#     # fix all conflict_fixable similarly
#     for c in range(C-1):
#         mask_c = conflict_fixable & (~used[c])
#         new_grid[mask_c] = c

#     # set vertex-cover nodes back to color C-1
#     for (i,j) in min_vc:
#         new_grid[i,j] = C-1

#     # 8) check convergence
#     num_conflicts = int((new_grid != grid).sum())
#     grid = new_grid
#     onehot = one_hot_encode(grid, C)

#     frac_conflicts = conflict_mask.float().mean().item() * 100
#     print(f"Iteration {iteration}: ~{frac_conflicts:.6f}% conflicted → resolved delta {num_conflicts}")

#     if num_conflicts == 0:
#         break

# # ----------------------
# # Finalize & Save
# # ----------------------
# final_coloring = grid.cpu().numpy()
# np.save(f"fixed_colorings/{args.run_id}_{gridsize}_fixed.npy", final_coloring)


# ----------------------
# Three‑phase fix: bonus→real, conflict→real, then single VC
# ----------------------

# Convert initial coloring to torch
grid = torch.from_numpy(grid_coloring).long().to(device)  # [H,W]
H, W = grid.shape
C = int(grid.max()) + 1  # number of colors including bonus

onehot = one_hot_encode(grid, C)  # [C,H,W]

# Phase 1: Greedy fix bonus cells → real
phase_start = perf_counter()
bonus_mask = (grid == C-1)            # [H,W]
used       = compute_used_colors(onehot, mask_kernel, pad)  # [C,H,W]

# find bonus cells with at least one free real color
free_in_bonus = bonus_mask & (~used[:C-1].any(dim=0))      # [H,W]

# assign each such cell the lowest‐index real color available
for c in range(C-1):
    can_use_c = free_in_bonus & (~used[c])  
    grid[can_use_c] = c

# recompute one‑hot
onehot = one_hot_encode(grid, C)
print(f"Phase 1 bonus greedy repair took {perf_counter() - phase_start:.2f}s", flush=True)

# Phase 2: Greedy fix real‑conflicting cells → real
phase_start = perf_counter()
neighbor_counts   = count_same_color_neighbors(onehot, mask_kernel, pad)  # [C,H,W]

# build 2D conflict mask
flat_colors       = grid.view(-1).long()               # [H*W]
flat_counts       = neighbor_counts.view(C, -1)        # [C, H*W]
flat_idx          = torch.arange(H*W, device=device)
selected_counts   = flat_counts[flat_colors, flat_idx] # [H*W]
conflict_mask     = (selected_counts.view(H, W) > 0)   # [H,W]

# recompute used-colors in this new grid from the neighbor counts above
used = neighbor_counts > 0   # [C,H,W]

free_in_conflict = conflict_mask & (~used[:C-1].any(dim=0))

for c in range(C-1):
    can_use_c = free_in_conflict & (~used[c])
    grid[can_use_c] = c

# recompute one‑hot
onehot = one_hot_encode(grid, C)
print(f"Phase 2 conflict greedy repair took {perf_counter() - phase_start:.2f}s", flush=True)


# Phase 3: Single VC on remaining conflicts
phase_start = perf_counter()
neighbor_counts = count_same_color_neighbors(onehot, mask_kernel, pad)  # [C,H,W]
flat_counts     = neighbor_counts.view(C, -1)
selected_counts = flat_counts[grid.view(-1), flat_idx]                 # [H*W]
conflict_mask   = (selected_counts.view(H, W) > 0)                    # [H,W]
print(f"Phase 3 conflict detection took {perf_counter() - phase_start:.2f}s", flush=True)

if args.skip_vertex_cover:
    print(
        f"Skipping vertex cover; leaving {int(conflict_mask.sum())} conflicted cells active for MILP",
        flush=True,
    )
else:
    # Build conflict-edge list by gathering from conflicted cells only.
    edge_start = perf_counter()
    conflict_pairs = build_conflict_pairs_sparse(grid, conflict_mask, mask_offsets)
    print(f"Built {len(conflict_pairs)} conflict-pair entries in {perf_counter() - edge_start:.2f}s", flush=True)

    # Compute vertex cover and recolor those to bonus
    vc_start = perf_counter()
    min_vc = graph_min_vertex_cover(conflict_pairs)
    for (i, j) in min_vc:
        grid[i, j] = C-1
    print(f"Computed vertex cover in {perf_counter() - vc_start:.2f}s", flush=True)

    print(f"Initially {conflict_mask.sum()} conflicts, fixed to bonus {len(min_vc)}")

    ### Recompute conflict mask

    onehot = one_hot_encode(grid, C)
    validation_start = perf_counter()
    neighbor_counts = count_same_color_neighbors(onehot, mask_kernel, pad)

    flat_colors     = grid.view(-1).long()
    flat_counts     = neighbor_counts.view(C, -1)
    flat_idx        = torch.arange(H*W, device=device)
    selected_counts = flat_counts[flat_colors, flat_idx]

    conflict_mask = (selected_counts.view(H, W) > 0)  # now up to date
    print(f"Final conflict validation took {perf_counter() - validation_start:.2f}s", flush=True)

# # Done!  No further fixes needed.
# final_coloring = grid.cpu()


# ----------------------
# Build and solve MILP
# ----------------------

# grid: a [H,W] torch.Tensor of initial colors (0..C-1), numpy if you prefer
# mask_offsets: an (M,2) numpy array of (di,dj) pairs
# H, W, C defined above

print(f"Now building integer program.")

# ----------------------
# Build solver with only “active” cells
# ----------------------
active_start = perf_counter()
active_mask = (grid == (C - 1)) | conflict_mask
active_coords = torch.nonzero(active_mask, as_tuple=False).cpu().numpy()
active = set(map(tuple, active_coords.tolist()))

print(f"{len(active)} active cells out of {H*W} in {perf_counter() - active_start:.2f}s")

mask_offsets_tuples = list(dict.fromkeys((int(di), int(dj)) for di, dj in mask_offsets))


def _compute_active_domains(active_lookup):
    domain_start = perf_counter()
    active_list = sorted(active_lookup)
    if not active_list:
        return {}

    active_coords = torch.tensor(active_list, device=grid.device, dtype=torch.long)
    ai = active_coords[:, 0]
    aj = active_coords[:, 1]

    active_mask_tensor = torch.zeros((H, W), dtype=torch.bool, device=grid.device)
    active_mask_tensor[ai, aj] = True
    forbidden = torch.zeros((len(active_list), C - 1), dtype=torch.bool, device=grid.device)
    offsets = torch.tensor(mask_offsets_tuples, device=grid.device, dtype=torch.long)

    chunk_size = 128
    for chunk_start in range(0, offsets.shape[0], chunk_size):
        offset_chunk = offsets[chunk_start:chunk_start + chunk_size]
        ni = (ai.unsqueeze(1) + offset_chunk[:, 0].unsqueeze(0)) % H
        nj = (aj.unsqueeze(1) + offset_chunk[:, 1].unsqueeze(0)) % W
        neighbour_is_frozen = ~active_mask_tensor[ni, nj]
        neighbour_colours = grid[ni, nj]
        real_frozen = neighbour_is_frozen & (neighbour_colours < C - 1)
        if not real_frozen.any():
            continue

        hit_nodes, hit_offsets = torch.nonzero(real_frozen, as_tuple=True)
        hit_colours = neighbour_colours[hit_nodes, hit_offsets].long()
        forbidden[hit_nodes, hit_colours] = True

    allowed = ~forbidden.cpu().numpy()
    domains = {
        node: tuple(int(c) for c in np.flatnonzero(allowed[idx]))
        for idx, node in enumerate(active_list)
    }

    real_variables = sum(len(domain) for domain in domains.values())
    forced_bonus = sum(len(domain) == 0 for domain in domains.values())
    print(
        "Feasible real-color domains: "
        f"{real_variables} real variables after pruning, "
        f"{forced_bonus} forced-bonus active cells "
        f"in {perf_counter() - domain_start:.2f}s"
    )
    return domains


def _shared_colours(left, right, domains):
    right_domain = set(domains[right])
    return tuple(c for c in domains[left] if c in right_domain)


def _extract_solution(component, x_vars, domains):
    chosen_by_node = {}
    for node in component:
        chosen = C - 1
        for c in domains[node]:
            value = pulp.value(x_vars[(node, c)])
            if value is not None and value > 0.5:
                chosen = c
                break
        chosen_by_node[node] = chosen
    return chosen_by_node


def _optional_bool(value):
    if value == 'default':
        return None
    return value == 'on'


def _apply_solver_tmp_dir(solver):
    """Force PuLP/CBC scratch files off the container's tiny /tmp mount."""
    tmp_dir = os.environ.get('PULP_TMP_DIR') or os.environ.get('TMPDIR')
    if tmp_dir and hasattr(solver, 'tmpDir'):
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        solver.tmpDir = tmp_dir
        print(f"Using solver tmpDir={tmp_dir}")
    return solver


def _make_mip_solver(msg):
    solver_threads = max(1, args.solver_threads)
    if args.solver_backend == 'cbc':
        return _apply_solver_tmp_dir(pulp.PULP_CBC_CMD(
            msg=msg,
            timeLimit=args.solver_time_limit,
            threads=solver_threads,
            gapAbs=args.solver_gap_abs,
            gapRel=args.solver_gap_rel,
            maxNodes=args.solver_max_nodes,
            cuts=_optional_bool(args.cbc_cuts),
            presolve=_optional_bool(args.cbc_presolve),
            strong=args.cbc_strong,
        ))
    if args.solver_backend == 'scip':
        return _apply_solver_tmp_dir(pulp.SCIP_CMD(
            path=args.scip_path,
            msg=msg,
            timeLimit=args.solver_time_limit,
            threads=solver_threads,
            gapAbs=args.solver_gap_abs,
            gapRel=args.solver_gap_rel,
            maxNodes=args.solver_max_nodes,
        ))
    raise ValueError(f"MIP solver requested for non-MIP backend {args.solver_backend!r}")


def _solve_global_milp(active_lookup, domains):
    print("Using global MILP over all active cells with domain pruning.")
    prob = pulp.LpProblem("HadwigerNelson", pulp.LpMinimize)
    x = {}
    b = {}

    print("Initializing variables...")
    for node in active_lookup:
        for c in domains[node]:
            x[(node, c)] = pulp.LpVariable(f"x_{node[0]}_{node[1]}_{c}", cat="Binary")
        b[node] = pulp.LpVariable(f"b_{node[0]}_{node[1]}", cat="Binary")

    print("Adding one-hot constraints...")
    for node in active_lookup:
        prob += (
            pulp.lpSum(x[(node, c)] for c in domains[node]) + b[node] == 1,
            f"assign_{node[0]}_{node[1]}",
        )

    print("Adding conflict constraints...")
    seen = set()
    reduced_edges = 0
    reduced_colour_constraints = 0
    for node in tqdm(active_lookup):
        i, j = node
        for di, dj in mask_offsets_tuples:
            neighbor = ((i + di) % H, (j + dj) % W)
            if neighbor == node:
                continue
            if neighbor in active_lookup:
                u, v = (node, neighbor) if node < neighbor else (neighbor, node)
                if (u, v) in seen:
                    continue
                seen.add((u, v))
                shared_colours = _shared_colours(u, v, domains)
                if not shared_colours:
                    continue
                reduced_edges += 1
                reduced_colour_constraints += len(shared_colours)
                for c in shared_colours:
                    prob += x[(u, c)] + x[(v, c)] <= 1
    print(
        "Reduced active-active constraints: "
        f"{reduced_edges} edges, {reduced_colour_constraints} color inequalities"
    )

    print("Initializing objective...")
    prob += pulp.lpSum(b[node] for node in active_lookup)

    print("Solving...")
    solver = _make_mip_solver(msg=True)
    prob.solve(solver)
    print(f" Status: {pulp.LpStatus[prob.status]}")

    return _extract_solution(active_lookup, x, domains)


def _build_component_variable_sets(component, domains):
    singleton_colour = {
        node: domains[node][0]
        for node in component
        if len(domains[node]) == 1
    }
    multi_nodes = [
        node
        for node in component
        if len(domains[node]) > 1
    ]
    explicit_real_variables = sum(len(domains[node]) for node in multi_nodes)
    return singleton_colour, multi_nodes, explicit_real_variables


def _connected_components(active_lookup, domains):
    active_list = sorted(active_lookup)
    if not active_list:
        return [], []

    parent = list(range(len(active_list)))
    rank = [0] * len(active_list)

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    active_coords = torch.tensor(active_list, device=grid.device, dtype=torch.long)
    ai = active_coords[:, 0]
    aj = active_coords[:, 1]

    active_index_grid = torch.full((H, W), -1, dtype=torch.long, device=grid.device)
    active_index_grid[ai, aj] = torch.arange(len(active_list), device=grid.device)

    allowed_np = np.zeros((len(active_list), C - 1), dtype=np.bool_)
    for idx, node in enumerate(active_list):
        domain = domains[node]
        if domain:
            allowed_np[idx, list(domain)] = True
    allowed = torch.from_numpy(allowed_np).to(device=grid.device)

    offsets = torch.tensor(mask_offsets_tuples, device=grid.device, dtype=torch.long)
    source_indices = torch.arange(len(active_list), device=grid.device).unsqueeze(1)
    edge_records = []
    reduced_edges = 0
    reduced_colour_constraints = 0
    edge_start = perf_counter()
    chunk_size = max(1, args.active_edge_chunk_size)
    print(
        "Building domain-aware active-cell connected components "
        f"with vectorized chunks of {chunk_size} offsets..."
    )
    for chunk_start in tqdm(range(0, offsets.shape[0], chunk_size)):
        offset_chunk = offsets[chunk_start:chunk_start + chunk_size]
        ni = (ai.unsqueeze(1) + offset_chunk[:, 0].unsqueeze(0)) % H
        nj = (aj.unsqueeze(1) + offset_chunk[:, 1].unsqueeze(0)) % W
        neighbor_indices = active_index_grid[ni, nj]
        candidate_mask = neighbor_indices > source_indices
        if not candidate_mask.any():
            continue

        left_idx, offset_idx = torch.nonzero(candidate_mask, as_tuple=True)
        right_idx = neighbor_indices[left_idx, offset_idx]
        shared = allowed[left_idx] & allowed[right_idx]
        has_shared = shared.any(dim=1)
        if not has_shared.any():
            continue

        left_idx = left_idx[has_shared].cpu().numpy()
        right_idx = right_idx[has_shared].cpu().numpy()
        shared_np = shared[has_shared].cpu().numpy()

        reduced_edges += int(len(left_idx))
        reduced_colour_constraints += int(shared_np.sum())
        for left, right, shared_row in zip(left_idx, right_idx, shared_np):
            left = int(left)
            right = int(right)
            shared_colours = tuple(int(c) for c in np.flatnonzero(shared_row))
            union(left, right)
            edge_records.append((left, right, shared_colours))

    components_by_root = defaultdict(list)
    for idx, node in enumerate(active_list):
        components_by_root[find(idx)].append(node)

    edges_by_root = defaultdict(list)
    for left_idx, right_idx, shared_colours in edge_records:
        root = find(left_idx)
        edges_by_root[root].append((
            active_list[left_idx],
            active_list[right_idx],
            shared_colours,
        ))

    print(
        "Reduced active-active constraints: "
        f"{reduced_edges} edges, {reduced_colour_constraints} color inequalities "
        f"in {perf_counter() - edge_start:.2f}s"
    )
    components = []
    component_edges = []
    for root, component in components_by_root.items():
        components.append(sorted(component))
        component_edges.append(edges_by_root.get(root, []))
    return components, component_edges


def _component_edges(component, component_lookup, domains):
    edges = set()
    for node in component:
        i, j = node
        for di, dj in mask_offsets_tuples:
            neighbor = ((i + di) % H, (j + dj) % W)
            if neighbor == node:
                continue
            if neighbor in component_lookup:
                u, v = (node, neighbor) if node < neighbor else (neighbor, node)
                shared_colours = _shared_colours(u, v, domains)
                if shared_colours:
                    edges.add((u, v, shared_colours))
    return edges


def _solve_component_milp(component, edges, domains, component_index):
    if not edges:
        return {
            node: (domains[node][0] if domains[node] else C - 1)
            for node in component
        }

    model_start = perf_counter()
    prob = pulp.LpProblem(f"HadwigerNelson_component_{component_index}", pulp.LpMinimize)
    x = {}
    b = {}
    singleton_colour, multi_nodes, explicit_real_variables = _build_component_variable_sets(
        component,
        domains,
    )
    print(
        f" Component {component_index}: {len(component)} nodes, "
        f"{len(singleton_colour)} singleton-domain nodes, "
        f"{explicit_real_variables} explicit real-color variables, "
        f"{len(edges)} active-active edges",
        flush=True,
    )

    for node in component:
        for c in domains[node] if len(domains[node]) > 1 else ():
            x[(node, c)] = pulp.LpVariable(
                f"x_{component_index}_{node[0]}_{node[1]}_{c}",
                cat="Binary",
            )
        b[node] = pulp.LpVariable(f"b_{component_index}_{node[0]}_{node[1]}", cat="Binary")

    for node in multi_nodes:
        prob += (
            pulp.lpSum(x[(node, c)] for c in domains[node]) + b[node] == 1,
            f"assign_{node[0]}_{node[1]}",
        )

    for edge_idx, (u, v, shared_colours) in enumerate(edges):
        for c in shared_colours:
            u_single = singleton_colour.get(u) == c
            v_single = singleton_colour.get(v) == c
            if u_single and v_single:
                prob += (
                    b[u] + b[v] >= 1,
                    f"edge_{edge_idx}_{c}",
                )
            elif u_single:
                prob += (
                    x[(v, c)] <= b[u],
                    f"edge_{edge_idx}_{c}",
                )
            elif v_single:
                prob += (
                    x[(u, c)] <= b[v],
                    f"edge_{edge_idx}_{c}",
                )
            else:
                prob += (
                    x[(u, c)] + x[(v, c)] <= 1,
                    f"edge_{edge_idx}_{c}",
                )

    prob += pulp.lpSum(b[node] for node in component)
    print(
        f" Component {component_index}: built compact MILP in "
        f"{perf_counter() - model_start:.2f}s",
        flush=True,
    )
    solver = _make_mip_solver(msg=args.component_solver_msg)
    solve_start = perf_counter()
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    objective_value = pulp.value(prob.objective)
    print(
        f" Component {component_index}: {args.solver_backend} status {status} in "
        f"{perf_counter() - solve_start:.2f}s, objective {objective_value}",
        flush=True,
    )
    if objective_value is None:
        raise RuntimeError(
            f"{args.solver_backend} did not return a feasible solution for component {component_index}."
        )
    if status != "Optimal":
        print(f" Component {component_index} solver status: {status}")

    chosen_by_node = {}
    for node in component:
        singleton = singleton_colour.get(node)
        if singleton is not None:
            value = pulp.value(b[node])
            chosen_by_node[node] = C - 1 if value is not None and value > 0.5 else singleton
            continue

        chosen = C - 1
        for c in domains[node]:
            value = pulp.value(x[(node, c)])
            if value is not None and value > 0.5:
                chosen = c
                break
        chosen_by_node[node] = chosen
    return chosen_by_node


def _solve_component_cp_sat(component, edges, domains, component_index):
    if not edges:
        return {
            node: (domains[node][0] if domains[node] else C - 1)
            for node in component
        }

    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError(
            "OR-Tools is required for --solver_backend=cp_sat. "
            "Install it with: uv pip install --python .venv/bin/python ortools"
        ) from exc

    model_start = perf_counter()
    cp = cp_model.CpModel()
    x = {}
    b = {}
    singleton_colour, multi_nodes, explicit_real_variables = _build_component_variable_sets(
        component,
        domains,
    )
    print(
        f" Component {component_index}: {len(component)} nodes, "
        f"{len(singleton_colour)} singleton-domain nodes, "
        f"{explicit_real_variables} explicit real-color variables, "
        f"{len(edges)} active-active edges",
        flush=True,
    )

    for node in component:
        for c in domains[node] if len(domains[node]) > 1 else ():
            x[(node, c)] = cp.NewBoolVar(f"x_{component_index}_{node[0]}_{node[1]}_{c}")
        b[node] = cp.NewBoolVar(f"b_{component_index}_{node[0]}_{node[1]}")

    for node in multi_nodes:
        cp.Add(sum(x[(node, c)] for c in domains[node]) + b[node] == 1)

    for u, v, shared_colours in edges:
        for c in shared_colours:
            u_single = singleton_colour.get(u) == c
            v_single = singleton_colour.get(v) == c
            if u_single and v_single:
                cp.Add(b[u] + b[v] >= 1)
            elif u_single:
                cp.Add(x[(v, c)] <= b[u])
            elif v_single:
                cp.Add(x[(u, c)] <= b[v])
            else:
                cp.Add(x[(u, c)] + x[(v, c)] <= 1)

    cp.Minimize(sum(b[node] for node in component))
    print(
        f" Component {component_index}: built CP-SAT model in "
        f"{perf_counter() - model_start:.2f}s",
        flush=True,
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(args.solver_time_limit)
    solver.parameters.num_search_workers = max(1, args.solver_threads)
    solver.parameters.log_search_progress = bool(args.component_solver_msg)
    if args.solver_gap_abs is not None:
        solver.parameters.absolute_gap_limit = float(args.solver_gap_abs)
    if args.solver_gap_rel is not None:
        solver.parameters.relative_gap_limit = float(args.solver_gap_rel)

    solve_start = perf_counter()
    status = solver.Solve(cp)
    status_name = solver.StatusName(status)
    print(
        f" Component {component_index}: cp_sat status {status_name} in "
        f"{perf_counter() - solve_start:.2f}s, objective {solver.ObjectiveValue()}",
        flush=True,
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT failed on component {component_index}: {status_name}")

    chosen_by_node = {}
    for node in component:
        singleton = singleton_colour.get(node)
        if singleton is not None:
            chosen_by_node[node] = C - 1 if solver.BooleanValue(b[node]) else singleton
            continue

        chosen = C - 1
        for c in domains[node]:
            if solver.BooleanValue(x[(node, c)]):
                chosen = c
                break
        chosen_by_node[node] = chosen
    return chosen_by_node


def _solve_component_milps(active_lookup, domains):
    forced_bonus = {
        node: C - 1
        for node in active_lookup
        if not domains[node]
    }
    active_with_real_domain = {
        node
        for node in active_lookup
        if domains[node]
    }
    if forced_bonus:
        print(f"{len(forced_bonus)} active cells have empty real-color domains and are forced to bonus.")

    components, component_edges = _connected_components(active_with_real_domain, domains)
    if not components:
        return forced_bonus

    sizes = [len(component) for component in components]
    print(
        "Connected components: "
        f"{len(components)} total, "
        f"{sum(size == 1 for size in sizes)} singletons, "
        f"max size {max(sizes)}"
    )

    chosen_by_node = dict(forced_bonus)
    total_edges = 0
    total_colour_constraints = 0
    for component_index, (component, edges) in enumerate(tqdm(
        zip(components, component_edges),
        total=len(components),
        desc="Solving components",
    )):
        total_edges += len(edges)
        total_colour_constraints += sum(len(shared_colours) for _, _, shared_colours in edges)
        if args.solver_backend == 'cp_sat':
            component_solution = _solve_component_cp_sat(component, edges, domains, component_index)
        else:
            component_solution = _solve_component_milp(component, edges, domains, component_index)
        chosen_by_node.update(
            component_solution
        )
    print(
        "Component MILPs covered "
        f"{total_edges} active-active edges and "
        f"{total_colour_constraints} color inequalities."
    )
    return chosen_by_node


active_domains = _compute_active_domains(active)
if args.no_component_decomposition:
    chosen_by_node = _solve_global_milp(active, active_domains)
else:
    chosen_by_node = _solve_component_milps(active, active_domains)

print("Extracting solution...")
opt_grid = grid.cpu().numpy().copy()
for (i, j), chosen in chosen_by_node.items():
    opt_grid[i, j] = chosen

# opt_grid now holds the IP-improved coloring


# Save & plot
fixed_colorings_dir = output_dir / "fixed_colorings"
fixed_colorings_dir.mkdir(parents=True, exist_ok=True)
np.save(fixed_colorings_dir / f"{args.run_id}_{gridsize}_IP_fixed.npy", opt_grid)

final_coloring = opt_grid

validation_start = perf_counter()
validation_grid = torch.from_numpy(final_coloring).long().to(device)
validation_onehot = one_hot_encode(validation_grid, C)
validation_counts = count_same_color_neighbors(validation_onehot, mask_kernel, pad)
validation_flat_counts = validation_counts.view(C, -1)
validation_flat_grid = validation_grid.view(-1)
validation_flat_idx = torch.arange(H * W, device=device)
validation_selected = validation_flat_counts[validation_flat_grid, validation_flat_idx]
real_conflict_cells = int(
    ((validation_flat_grid < C - 1) & (validation_selected > 0)).sum().item()
)
print(
    f"Final real-color conflict cells: {real_conflict_cells} "
    f"in {perf_counter() - validation_start:.2f}s",
    flush=True,
)
if real_conflict_cells:
    raise RuntimeError(
        f"MILP solution still has {real_conflict_cells} real-color conflict cells."
    )

# Optional: report fraction bonus
count_bonus = np.count_nonzero(final_coloring == (C-1))
frac_bonus  = count_bonus / (H*W) * 100
print(f"Fraction set to bonus color: {frac_bonus:.8f}%")


# final_coloring = grid
# print(f"{final_coloring.shape=}")
# print(f"{final_coloring.size()=}")
# print(f"{final_coloring.dtype=}")
# print(f"{type(final_coloring)=}")
# print(f"{(final_coloring == 5)}")
print(f"Fraction set to 5: {(final_coloring == 5).sum() / (H * W) * 100:.8f}%")
# np.save(f"fixed_colorings/{args.run_id}_{gridsize}_fixed", final_coloring.cpu().numpy())

import os
import csv

def save_fixed_fraction(run_id, eval_gridsize, fraction_fixed, csv_filename='verified_paralellograms_ip.csv'):
    file_exists = os.path.isfile(csv_filename)
    data = {
        'run_id': run_id,
        'eval_gridsize': eval_gridsize,
        'fraction_fixed_to_5 (%)': round(float(fraction_fixed) * 100, 8)
    }

    with open(csv_filename, mode='a', newline='') as csvfile:
        fieldnames = ['run_id', 'eval_gridsize', 'fraction_fixed_to_5 (%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

fraction_fixed = (final_coloring == 5).sum() / (H * W)
save_fixed_fraction(args.run_id, gridsize, fraction_fixed, csv_filename=str(output_dir / 'verified_paralellograms_ip.csv'))

if not args.no_plot:
    plot_filename = fixed_colorings_dir / f"{args.run_id}_{gridsize}_fixed.png" if H > 100 else fixed_colorings_dir / f"{args.run_id}_{gridsize}_fixed.pdf"

    plot_parallelogram_coloring(starts_reshaped, final_coloring, parallelogram[0], parallelogram[1], gridsize, plot_filename)
