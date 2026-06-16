import argparse
import json
import os
import sys
from pathlib import Path
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
args = parser.parse_args()

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
grid_bounds = (3, 3)

grid_colours, conflicts_per_point, _ = GeneralUtility.evaluate_on_grid(
    grid_bounds=grid_bounds,
    model=model,
    device=device,
    n_circle_points=1,
    gridsize=gridsize,
    dim=2,
    p_norm=2,
    concat_colours=False,
    colour_distances=[1]*n_colours,
    good_coloring=True,
    verbose=False
)

coloring = grid_colours.detach().cpu().numpy()


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

def compute_base_mask_vec(parallelogram, N, radius=1.0, device='cpu'):
    v1, v2 = parallelogram[0], parallelogram[1]
    # 1) Fundamental starts: (N^2 x 2)
    lin = torch.linspace(0, 1, N+1, device=device)[:-1]
    a, b = torch.meshgrid(lin, lin, indexing='ij')
    starts = (a.reshape(-1, 1)*v1 + b.reshape(-1, 1)*v2)  # [N^2, 2]

    # 2) Tile offsets: (9 x 2)
    offs = torch.stack(torch.meshgrid(torch.tensor([-1,0,1], device=device),
                                      torch.tensor([-1,0,1], device=device),
                                      indexing='ij'), -1).view(-1, 2)
    tile_shifts = offs[:,0:1]*v1 + offs[:,1:2]*v2          # [9, 2]

    # 3) All shifted starts: (9*N^2 x 2)
    shifted = starts.unsqueeze(0) + tile_shifts.unsqueeze(1)   # [9, N^2, 2]
    shifted = shifted.view(-1, 2)                             # [9*N^2, 2]

    # 4) Compute corners for each micro‑square: (9*N^2 x 4 x 2)
    dv1, dv2 = v1/N, v2/N
    # corner offsets relative to each start:
    corner_offs = torch.stack([
        torch.zeros(2, device=device),
        dv1,
        dv1 + dv2,
        dv2
    ], dim=0)  # [4,2]
    corners = shifted.unsqueeze(1) + corner_offs.unsqueeze(0)  # [9*N^2,4,2]

    # 5) Centers: same set of the 4 positions, shape [4,2]
    centers = corner_offs  # reuse the same

    # 6) Compute distance²: broadcast to [9*N^2,4,4]
    #   dims: tiles*points × corners × centers
    diffs = corners.unsqueeze(2) - centers.view(1,1,4,2)       # [9*N²,4(c),4(cent),2]
    d2 = (diffs**2).sum(dim=-1)                                # [9*N²,4,4]

    # 7) inside mask: [9*N²,4,4] bool
    inside = d2 <= radius*radius

    # 8) for each micro‑square & each center, do “any but not all” over corners
    any_inside = inside.any(dim=1)   # [9*N²,4]
    all_inside = inside.all(dim=1)   # [9*N²,4]
    partial = (any_inside & (~all_inside)).any(dim=1)  # [9*N²]

    # 9) pick out the partial ones
    idxs = torch.nonzero(partial, as_tuple=False).squeeze(1) # indices in [0..9*N²)
    
    # 10) Map back to fundamental (i,j):
    #    Each block of N² belongs to one tile; floor_divide by N² to get tile#
    #    idx % N² => cell index within original grid
    cell_idx = idxs % (N*N)
    i = cell_idx // N
    j = cell_idx % N

    base_mask = torch.stack([i, j], dim=1).tolist()
    return starts.cpu(), base_mask

#starts, base_mask = compute_base_mask(parallelogram.cpu(), gridsize)
starts, base_mask = compute_base_mask_vec(parallelogram, gridsize, device=device)

starts_reshaped = starts.reshape(gridsize, gridsize, -1)
with torch.no_grad():
    outputs = model(starts.to(device)).argmax(dim=1).cpu()
    grid_coloring = outputs.reshape(gridsize, gridsize).numpy()


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

def count_same_color_neighbors(onehot: torch.Tensor,
                               mask_kernel: torch.Tensor,
                               pad: tuple) -> torch.Tensor:
    """
    onehot: [C,H,W] float
    mask_kernel: [1,1,Kh,Kw]
    pad: (left, right, top, bottom)
    Returns: [C,H,W] int counts of same-color neighbors.
    """
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
    mask_kernel: [1,1,Kh,Kw]
    pad: (left, right, top, bottom)
    Returns: [C,H,W] bool indicating if any neighbor has that color.
    """
    C, H, W = onehot.shape

    # 1) pad the tensor on the torus
    x = F.pad(onehot.unsqueeze(0), pad=pad, mode='circular')  # [1,C,H',W']

    # 2) replicate the mask kernel so there's one per input channel
    weight = mask_kernel.repeat(C, 1, 1, 1)                   # [C,1,Kh,Kw]

    # 3) grouped convolution: each channel is convolved with its own copy of mask_kernel
    counts = F.conv2d(x, weight=weight, groups=C)             # [1,C,H,W]

    # 4) drop batch dim and threshold
    return (counts.squeeze(0) > 0)                           # [C,H,W] bool


def graph_min_vertex_cover(conflict_pairs):
    """
    Given list of edges (u,v) in pixel‑coordinate space, build graph and return approx vertex cover.
    """
    G = nx.Graph()
    G.add_edges_from(conflict_pairs)
    return nx.algorithms.approximation.min_weighted_vertex_cover(G)

# ----------------------
# Build static data
# ----------------------
# base_mask is a list of (i,j) offsets
_, base_mask = compute_base_mask_vec(parallelogram, gridsize, device=device)
mask_offsets = np.array(base_mask, dtype=int)

print(f"Computed base mask. {len(base_mask)=}, {base_mask[0]=}, mask_offsets={mask_offsets.shape=}")

# build convolution kernel once
mask_kernel, pad = build_mask_kernel(mask_offsets, device)

print(f"Computed mask kernel. {mask_kernel.shape=}, pad={pad}")

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

# Phase 2: Greedy fix real‑conflicting cells → real
neighbor_counts   = count_same_color_neighbors(onehot, mask_kernel, pad)  # [C,H,W]

# build 2D conflict mask
flat_colors       = grid.view(-1).long()               # [H*W]
flat_counts       = neighbor_counts.view(C, -1)        # [C, H*W]
flat_idx          = torch.arange(H*W, device=device)
selected_counts   = flat_counts[flat_colors, flat_idx] # [H*W]
conflict_mask     = (selected_counts.view(H, W) > 0)   # [H,W]

# recompute used‑colors in this new grid
used = compute_used_colors(onehot, mask_kernel, pad)   # [C,H,W]

free_in_conflict = conflict_mask & (~used[:C-1].any(dim=0))

for c in range(C-1):
    can_use_c = free_in_conflict & (~used[c])
    grid[can_use_c] = c

# recompute one‑hot
onehot = one_hot_encode(grid, C)


# Phase 3: Single VC on remaining conflicts
neighbor_counts = count_same_color_neighbors(onehot, mask_kernel, pad)  # [C,H,W]
flat_counts     = neighbor_counts.view(C, -1)
selected_counts = flat_counts[grid.view(-1), flat_idx]                 # [H*W]
conflict_mask   = (selected_counts.view(H, W) > 0)                    # [H,W]

# Build conflict‐edge list (vectorized via roll)
conflict_pairs = []
cm = conflict_mask
for di, dj in mask_offsets:
    rolled = torch.roll(grid, shifts=(-di, -dj), dims=(0, 1))
    match  = cm & (rolled == grid)
    if not match.any():
        continue
    idxs = torch.nonzero(match, as_tuple=False)
    for i, j in idxs.tolist():
        ni, nj = (i + di) % H, (j + dj) % W
        conflict_pairs.append(((i, j), (ni, nj)))

# Compute vertex cover and recolor those to bonus
min_vc = graph_min_vertex_cover(conflict_pairs)
for (i, j) in min_vc:
    grid[i, j] = C-1

print(f"Initially {conflict_mask.sum()} conflicts, fixed to bonus {len(min_vc)}")

### Recompute conflict mask

onehot = one_hot_encode(grid, C)
neighbor_counts = count_same_color_neighbors(onehot, mask_kernel, pad)

flat_colors     = grid.view(-1).long()
flat_counts     = neighbor_counts.view(C, -1)
flat_idx        = torch.arange(H*W, device=device)
selected_counts = flat_counts[flat_colors, flat_idx]

conflict_mask = (selected_counts.view(H, W) > 0)  # now up to date

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
prob = pulp.LpProblem("HadwigerNelson", pulp.LpMinimize)

x = {}   # x[(i,j,c)]
b = {}   # b[(i,j)]

# 1) Identify the set of “active” cells:
#    these are either bonus‑colored after greedy or still in conflict_mask
active = set()
# assuming `grid` is your post‑greedy 2D torch.Tensor [H,W]
# and `conflict_mask` is your [H,W] bool tensor from the last pass:

for i in range(H):
    for j in range(W):
        if grid[i,j].item() == (C-1) or conflict_mask[i,j].item():
            active.add((i,j))

print(f"{len(active)} active cells out of {H*W}")

# 2) Create variables only for those active cells
print("Initializing variables...")
for (i,j) in active:
    # real‐color binaries
    for c in range(C-1):
        x[(i,j,c)] = pulp.LpVariable(f"x_{i}_{j}_{c}", cat="Binary")
    # bonus binary
    b[(i,j)]   = pulp.LpVariable(f"b_{i}_{j}", cat="Binary")

# 3) One‑hot constraints only for active cells
print("Adding one-hot constraints...")
for (i,j) in active:
    prob += (
        pulp.lpSum(x[(i,j,c)] for c in range(C-1)) + b[(i,j)]
        == 1,
        f"assign_{i}_{j}"
    )

print("Adding conflict constraints...")

seen = set()
for (i,j) in tqdm(active):
    for di, dj in mask_offsets:
        ni, nj = (i+di) % H, (j+dj) % W

        if (ni, nj) in active:
            # --- Case A: active ↔ active, add pairwise constraints once ---
            u, v = (i,j), (ni,nj)
            if v < u:
                u, v = v, u
            if (u,v) in seen:
                continue
            seen.add((u,v))

            for c in range(C-1):
                prob += x[(u[0],u[1],c)] + x[(v[0],v[1],c)] <= 1

        else:
            # --- Case B: active ↔ frozen neighbor at (ni,nj) ---
            # If frozen neighbor has a real color c*, forbid x_{i,j,c*}=1
            cstar = int(grid[ni, nj].item())
            if cstar < C-1:  
                # only enforce if neighbor is real‐colored
                prob += x[(i,j,cstar)] <= 0
            # if neighbor is bonus (cstar==C-1), no constraint needed

# 5) Objective: minimize bonus among active cells
print("Initializing objective...")
prob += pulp.lpSum(b[(i,j)] for (i,j) in active)

# 6) Solve
solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=360000)
print("Solving...")
prob.solve(solver)
print(f" Status: {pulp.LpStatus[prob.status]}")

# 7) Extract solution: start from frozen greedy grid, overwrite actives
print("Extracting solution...")
opt_grid = grid.cpu().numpy().copy()
for (i,j) in active:
    # default to bonus if no x==1
    chosen = C-1
    for c in range(C-1):
        if pulp.value(x[(i,j,c)]) > 0.5:
            chosen = c
            break
    opt_grid[i,j] = chosen

# opt_grid now holds the IP‑improved coloring


# Save & plot
os.makedirs("fixed_colorings", exist_ok=True)
np.save(f"fixed_colorings/{args.run_id}_{gridsize}_IP_fixed.npy", opt_grid)

final_coloring = opt_grid

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
print(f"Fraction set to 5: {(final_coloring == 5).sum() / args.eval_gridsize**2 * 100:.8f}%")
# np.save(f"fixed_colorings/{args.run_id}_{gridsize}_fixed", final_coloring.cpu().numpy())

import os
import csv

def save_fixed_fraction(run_id, eval_gridsize, fraction_fixed, csv_filename='verified_paralellograms_ip.csv'):
    file_exists = os.path.isfile(csv_filename)
    data = {
        'run_id': run_id,
        'eval_gridsize': eval_gridsize,
        'fraction_fixed_to_5 (%)': round(fraction_fixed.item() * 100, 8)
    }

    with open(csv_filename, mode='a', newline='') as csvfile:
        fieldnames = ['run_id', 'eval_gridsize', 'fraction_fixed_to_5 (%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

fraction_fixed = (final_coloring == 5).sum() / args.eval_gridsize**2
save_fixed_fraction(args.run_id, gridsize, fraction_fixed)

plot_filename = f"fixed_colorings/{args.run_id}_{gridsize}_fixed.png" if H > 100 else f"fixed_colorings/{args.run_id}_{gridsize}_fixed.pdf"

plot_parallelogram_coloring(starts_reshaped, final_coloring, parallelogram[0], parallelogram[1], gridsize, plot_filename)