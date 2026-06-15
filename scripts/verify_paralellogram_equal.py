import argparse
import math
from pathlib import Path
import sys
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import wandb
import networkx as nx
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
sys.path.append("..")
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from models import ResMLP
from utilities import GeneralUtility

# ----------------------
# Argument Parsing
# ----------------------

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=str, default = "c04fwjda")
parser.add_argument('--max_size', type=float, default=0.01)
args = parser.parse_args()

run_id = args.run_id

# ----------------------
# Load W&B Run
# ----------------------
api = wandb.Api()
run = api.run(f"ais2t/2DHadwigerNelson/{run_id}")

# try:
#     checkpoint_file = 'step_32768_model.pt'
#     ckpt = run.file(checkpoint_file).download(root=f"./models/{run_id}", replace=True)
# except:
#     checkpoint_file = 'step_65536_model.pt'
#     ckpt = run.file(checkpoint_file).download(root=f"./models/{run_id}", replace=True)

checkpoint_file = 'trained_model.pt'
ckpt = run.file(checkpoint_file).download(root=f"./models/{run_id}", replace=True)


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

    if run.config['n_colours'] == 7:
        colors = [
      '#FFD6A5',  # light orange"
      '#FDFFB6',  # light yellow
      '#CAFFBF',  # light green
      '#9BF6FF',  # light turquoise
      '#A0C4FF',  # light blue
      '#FFADAD',  # light red
      '#000000',  # black
   ]

    
    pastel_cmap = ListedColormap(colors, name='pastel')

    return pastel_cmap

def plot_parallelogram_coloring(starts, values, v1, v2, max_size, filename=None):
    """
    starts:     (N1*N2,2) tensor of fundamental cell lower-lefts
    values:     (N1*N2,) tensor of color-indices (or floats)
    v1,v2:      torch tensors, shape (2,)
    max_size:   maximum allowed side-length of each micro-parallelogram
    mask:       list of (i,j) index pairs (in the N1×N2 grid) to outline
    filename:   if provided, save to disk; otherwise plt.show()
    """
    # 1) recompute subdivisions so each side ≤ max_size
    L1 = v1.norm().item()
    L2 = v2.norm().item()
    N1 = max(1, math.ceil(L1 / max_size))
    N2 = max(1, math.ceil(L2 / max_size))

    # 2) prep data as numpy
    starts_np = starts.cpu().reshape(N1 * N2, 2).numpy()
    values_np = values.flatten()
    dv1 = v1.cpu().numpy() / N1
    dv2 = v2.cpu().numpy() / N2

    # 3) colormap setup (assumes last_color_idx defined globally or adapt as needed)
    cmap = get_cmap()
    # if your values are integer class-IDs 0..K, set last_color_idx = K
    # otherwise you can normalize by values_np.max()
    last_color_idx = float(values_np.max())

    # 4) draw all micro–parallelograms colored by `values`
    fig, ax = plt.subplots(figsize=(8, 8))
    for p, val in tqdm(zip(starts_np, values_np), total=N1 * N2):
        corners = np.stack([
            p,
            p + dv1,
            p + dv1 + dv2,
            p + dv2
        ])
        poly = Polygon(corners,
                       closed=True,
                       facecolor=cmap(val / last_color_idx),
                       edgecolor='none')
        ax.add_patch(poly)


    # 6) set axis limits so we see the whole tiling
    min_x, min_y = starts_np.min(axis=0)
    max_x, max_y = starts_np.max(axis=0)
    lo = min(min_x, min_y)
    hi = max(max_x, max_y)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.axis('off')

    # 7) draw v1, v2 arrows + length labels
    ax.arrow(0, 0, v1[0].item(), v1[1].item(),
             head_width=0.05, head_length=0.2, fc='black', ec='black')
    ax.arrow(0, 0, v2[0].item(), v2[1].item(),
             head_width=0.05, head_length=0.2, fc='black', ec='black')
    ax.text(v1[0].item()/2, v1[1].item()/2,
            f"{v1.norm().item():.2f}", fontsize=12,
            ha='center', va='center', color='green')
    ax.text(v2[0].item()/2, v2[1].item()/2,
            f"{v2.norm().item():.2f}", fontsize=12,
            ha='center', va='center', color='green')

    # 9) finish
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()
# ----------------------
# Load Model
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim = run.config["dim"]

model = ResMLP(input_dim=input_dim, output_dim=run.config['n_colours'], device=device, **run.config["model"])
model.load_state_dict(torch.load(ckpt.name, map_location=device))
model = model.to(device).eval()

if run.config["training"].get("trainable_parallelogram", False):
    v00 = run.summary.parallelogram_0_0
    v01 = run.summary.parallelogram_0_1
    v10 = run.summary.parallelogram_1_0
    v11 = run.summary.parallelogram_1_1
    parallelogram = torch.tensor([[v00, v01], [v10, v11]], device=device)
    print(f"Parallelogram: {parallelogram}")

else:
    parallelogram = torch.tensor(run.config["training"]["parallelogram"], device=device)
model = GeneralUtility.prepend_parallelogram_transformation(model, spanning_vectors=parallelogram)

# ----------------------
# Evaluate Model
# ----------------------
max_size = args.max_size
n_colours = run.config['n_colours']
last_color_idx = n_colours - 1
grid_bounds = (3, 3)

L1 = parallelogram[0].norm().item()
L2 = parallelogram[1].norm().item()
N1 = max(1, math.ceil(L1 / max_size))
N2 = max(1, math.ceil(L2 / max_size))

print(f"Grid size: {N1} x {N2}")


# ----------------------
# Compute Base Mask (once)
# ----------------------
def compute_base_mask_vec(parallelogram, max_size, radius=1.0, device='cpu'):
    v1, v2 = parallelogram[0].to(device), parallelogram[1].to(device)
    # ------------------------------------------------------------
    # 0) decide subdivisions so each small parallelogram side ≤ max_size
    L1 = v1.norm().item()
    L2 = v2.norm().item()
    N1 = max(1, math.ceil(L1 / max_size))
    N2 = max(1, math.ceil(L2 / max_size))
    # ------------------------------------------------------------
    # 1) Fundamental starts: (N1*N2 × 2)
    lin1 = torch.linspace(0, 1, N1+1, device=device)[:-1]
    lin2 = torch.linspace(0, 1, N2+1, device=device)[:-1]
    a, b = torch.meshgrid(lin1, lin2, indexing='ij')   # shape N1×N2
    starts = (a.reshape(-1,1)*v1 + b.reshape(-1,1)*v2) # [N1*N2, 2]

    # 2) Tile offsets: same 3×3 tiling
    offs = torch.stack(
        torch.meshgrid(
            torch.tensor([-1,0,1], device=device),
            torch.tensor([-1,0,1], device=device),
            indexing='ij'
        ), -1
    ).view(-1,2)                            # [9,2]
    tile_shifts = offs[:,0:1]*v1 + offs[:,1:2]*v2  # [9,2]

    # 3) All shifted starts: [9*N1*N2, 2]
    shifted = (starts.unsqueeze(0) + tile_shifts.unsqueeze(1)).view(-1,2)

    # 4) Corners for each micro-parallelogram
    dv1, dv2 = v1 / N1, v2 / N2
    corner_offs = torch.stack([
        torch.zeros(2, device=device),
        dv1,
        dv1 + dv2,
        dv2
    ], dim=0)                             # [4,2]
    corners = shifted.unsqueeze(1) + corner_offs.unsqueeze(0)  # [9*N1*N2,4,2]

    # 5) Center points (same offsets)
    centers = corner_offs                        # [4,2]

    # 6) Squared distances [9*N1*N2,4(corners),4(centers)]
    diffs = corners.unsqueeze(2) - centers.view(1,1,4,2)
    d2 = (diffs**2).sum(dim=-1)

    # 7–8) Find “partial” cells (any inside but not all)
    inside      = d2 <= radius*radius         # [9*N1*N2,4,4]
    any_inside  = inside.any(dim=1)            # [9*N1*N2,4]
    all_inside  = inside.all(dim=1)            # [9*N1*N2,4]
    partial     = (any_inside & (~all_inside)).any(dim=1)

    # 9–10) collect their (i,j) in the fundamental N1×N2 grid
    idxs     = torch.nonzero(partial, as_tuple=False).squeeze(1)
    cell_idx = idxs % (N1 * N2)
    i = cell_idx // N2
    j = cell_idx % N2

    base_mask = torch.stack([i, j], dim=1).tolist()
    return starts.cpu(), base_mask

#starts, base_mask = compute_base_mask(parallelogram.cpu(), gridsize)
starts, base_mask = compute_base_mask_vec(parallelogram, max_size, device=device)

with torch.no_grad():
    outputs = model(starts.to(device)).argmax(dim=1).cpu()
    grid_coloring = outputs.reshape(N1, N2).numpy().astype(np.int8)

starts_reshaped = starts.reshape(N1, N2, 2)
if max(N1, N2) < 200:
    plot_parallelogram_coloring(starts = starts_reshaped,
                                values = grid_coloring, 
                                v1 = parallelogram[0], 
                                v2 = parallelogram[1], 
                                max_size = max_size, 
                                filename = f"equal_parallelograms/{run_id}_{max_size}_initial.pdf")

import torch
from multiprocessing import Pool
from tqdm import tqdm

# --------------------------------------------------
# Conflict Detection for a rectangular grid
# --------------------------------------------------
# Assumes the following globals are defined at module scope:
#   last_color_idx: integer index of the "blank" color (exclusive)

# Worker init no longer needs last_color_idx because it's inherited
# via module globals when using fork-based Pool.
def init_worker(grid_coloring, mask):
    global global_coloring, global_mask, global_n1, global_n2
    global_coloring = grid_coloring
    global_mask     = mask
    # Treat grid_coloring as shape (N1 rows, N2 cols)
    global_n1, global_n2 = grid_coloring.shape

# ----------------------------------------------------------------
# Process a single cell to detect conflicts and used colors
# ----------------------------------------------------------------
def process_center(center):
    """
    center: (i, j) with i in [0..N1-1], j in [0..N2-1]
    Returns tuple:
        cc        = color at (i,j)
        unused    = list of unused color indices around (i,j)
        conflicts = list of neighbor coords with same non-blank color
    """
    i, j = center
    cc = int(global_coloring[i, j].item())

    conflicts   = []
    used_colors = set()

    for di, dj in global_mask:
        ni = (i + int(di)) % global_n1
        nj = (j + int(dj)) % global_n2
        nc = int(global_coloring[ni, nj].item())

        # same non-blank color => conflict
        if nc == cc and nc != last_color_idx:
            conflicts.append((ni, nj))
        # collect used (non-blank) colors
        if nc != last_color_idx:
            used_colors.add(nc)

    # real colors are in [0..last_color_idx-1]
    unused = [k for k in range(last_color_idx) if k not in used_colors]
    return cc, unused, conflicts

# ----------------------------------------------------------------
# Parallel conflict detection over entire grid
# ----------------------------------------------------------------
def process_all_centers(grid_coloring, mask, num_workers=4):
    """
    Runs process_center on every cell in parallel.

    Args:
      grid_coloring: 2D torch tensor of shape (N1×N2)
      mask:          list of (di, dj)
      num_workers:   number of parallel processes

    Returns:
      centers: list of (i,j)
      results: list of (cc, unused, conflicts)
    """
    N1, N2 = grid_coloring.shape
    centers = [(i, j) for i in range(N1) for j in range(N2)]

    with Pool(
        processes   = num_workers,
        initializer = init_worker,
        initargs    = (grid_coloring, mask)
    ) as pool:
        results = list(
            tqdm(
                pool.imap(process_center, centers),
                total=len(centers)
            )
        )

    return centers, results




# ----------------------
# Conflict Analysis
# ----------------------
def analyze_conflicts(results, centers, grid_coloring):
    five_non_fixable = []
    five_fixable = []
    conflict_non_fixable = {}
    conflict_fixable = []

    for center, (cc, unused_colors, conflicts) in zip(centers, results):
        if cc == last_color_idx:
            if len(unused_colors) > 0:
                five_fixable.append(center)
            else:
                five_non_fixable.append(center)
        else:
            if len(conflicts) > 0:
                if any(c != cc for c in unused_colors):
                    conflict_fixable.append(center)
                else:
                    conflict_non_fixable[center] = conflicts

    G = nx.Graph()
    for center, conflicts in conflict_non_fixable.items():
        G.add_node(center)
        for conflict in conflicts:
            G.add_edge(center, conflict)

    min_vertex_cover = nx.algorithms.approximation.min_weighted_vertex_cover(G)

    percent_remaining = (len(five_non_fixable) + len(min_vertex_cover)) / (grid_coloring.shape[0]**2) * 100

    return {
        "five_non_fixable": five_non_fixable,
        "five_fixable": five_fixable,
        "conflict_non_fixable": conflict_non_fixable,
        "conflict_fixable": conflict_fixable,
        "min_vertex_cover": min_vertex_cover,
        "percent_remaining_conflicts": percent_remaining,
    }

# ----------------------
# Fix Coloring
# ----------------------
# ----------------------------------------------------------------
# Fix coloring based on detected issues
# ----------------------------------------------------------------
def fix_coloring(grid_coloring, mask, five_fixable, conflict_fixable, min_vertex_cover):
    """
    Adjusts grid_coloring to resolve easy conflicts:
      - five_fixable:      list of centers only touching ≤4 neighbors
      - conflict_fixable:  list of centers with simple color conflicts
      - min_vertex_cover:  centers to force-blank (last_color_idx)

    Returns a numpy array of the fixed coloring.
    """
    # Copy input to numpy
    if isinstance(grid_coloring, torch.Tensor):
        grid_np = grid_coloring.cpu().numpy()
    else:
        grid_np = np.array(grid_coloring, copy=True)

    fixed = grid_np.copy()
    # Treat grid as (N1 rows, N2 cols)
    N1, N2 = grid_np.shape

    # 1) Fix five-fixable centers by assigning first unused color
    for (i, j) in five_fixable:
        used_colors = set()
        for di, dj in mask:
            ni = (i + di) % N1
            nj = (j + dj) % N2
            mc = grid_np[ni, nj]
            if mc != last_color_idx:
                used_colors.add(int(mc))
        unused = [k for k in range(last_color_idx) if k not in used_colors]
        if unused:
            fixed[i, j] = unused[0]

    # 2) Fix conflict-fixable centers similarly, but use updated neighbors
    for (i, j) in conflict_fixable:
        used_colors = set()
        for di, dj in mask:
            ni = (i + di) % N1
            nj = (j + dj) % N2
            mc = fixed[ni, nj]
            if mc != last_color_idx:
                used_colors.add(int(mc))
        unused = [k for k in range(last_color_idx) if k not in used_colors]
        if unused:
            fixed[i, j] = unused[0]

    # 3) Force-blank all in min-vertex-cover
    for (i, j) in min_vertex_cover:
        fixed[i, j] = last_color_idx

    return fixed


def fix_coloring_trivial(grid_coloring, mask, five_fixable, conflict_fixable, min_vertex_cover):
    fixed_coloring = np.copy(grid_coloring)

    for center in five_fixable:
        fixed_coloring[center[0], center[1]] = last_color_idx 

    for center in conflict_fixable:
        fixed_coloring[center[0], center[1]] = last_color_idx

    for center in min_vertex_cover:
        fixed_coloring[center[0], center[1]] = last_color_idx

    return fixed_coloring

# ----------------------
# Iterative Fixing
# ----------------------




centers, results = process_all_centers(grid_coloring, base_mask)
stats = analyze_conflicts(results, centers, grid_coloring)

count = 0

total_bad = len(stats["conflict_non_fixable"]) + len(stats["conflict_fixable"]) + len(stats["five_non_fixable"]) + len(stats["five_fixable"])

print(f"Initial stats: \n Non-fixable conflicts: {len(stats['conflict_non_fixable'])} ({len(stats['conflict_non_fixable']) / (N1*N2) * 100:.8f}%)\n Fixable conflicts: {len(stats['conflict_fixable'])} ({len(stats['conflict_fixable']) / (N1*N2) * 100:.8f}%)\n Non-fixable five: {len(stats['five_non_fixable'])} ({len(stats['five_non_fixable']) / (N1*N2) * 100:.8f}%)\n Fixable five: {len(stats['five_fixable'])} ({len(stats['five_fixable']) / (N1*N2) * 100:.8f}%)\n Total bad: {total_bad} ({total_bad / (N1*N2) * 100:.8f}%)\n")

fixed_coloring = fix_coloring(grid_coloring, base_mask, stats["five_fixable"], stats["conflict_fixable"], stats["min_vertex_cover"])

if max(N1, N2) < 500:

    initial_fix = fix_coloring_trivial(grid_coloring, base_mask, stats["five_fixable"], stats["conflict_fixable"], stats["min_vertex_cover"])
    if max(N1, N2) <= 300: 
        suffix = ".pdf"
    else:
        suffix = ".png"

    starts_reshaped = starts.reshape(N1, N2, 2)
    plot_parallelogram_coloring(starts_reshaped, initial_fix, parallelogram[0], parallelogram[1], max_size, f"equal_parallelograms/{run_id}_{max_size}_trivial" + suffix)




next_iteration = fixed_coloring


while len(stats["conflict_fixable"]) + len(stats["conflict_non_fixable"]) + len(stats["five_fixable"]) > 0:


    count += 1
    next_iteration = fix_coloring(next_iteration, base_mask, stats["five_fixable"], stats["conflict_fixable"], stats["min_vertex_cover"])
    centers, results = process_all_centers(next_iteration, base_mask)
    stats = analyze_conflicts(results, centers, next_iteration)

    total_bad = len(stats["conflict_non_fixable"]) + len(stats["conflict_fixable"]) + len(stats["five_non_fixable"]) + len(stats["five_fixable"])

    print(f"Iteration {count}: \n Non-fixable conflicts: {len(stats['conflict_non_fixable'])} ({len(stats['conflict_non_fixable']) / (N1*N2) * 100:.8f}%)\n Fixable conflicts: {len(stats['conflict_fixable'])} ({len(stats['conflict_fixable']) / (N1*N2) * 100:.8f}%)\n Non-fixable five: {len(stats['five_non_fixable'])} ({len(stats['five_non_fixable']) / (N1*N2) * 100:.8f}%)\n Fixable five: {len(stats['five_fixable'])} ({len(stats['five_fixable']) / (N1*N2) * 100:.8f}%)\n Total bad: {total_bad} ({total_bad / (N1*N2) * 100:.8f}%)\n")

    if count > 10:
        for group in ["five_non_fixable", "conflict_non_fixable", "five_fixable", "conflict_fixable"]:
            for center in stats[group]:
                next_iteration[center[0], center[1]] = last_color_idx
        break

print(f"Converged after {count} iterations.")
final_coloring = next_iteration
print(f"Fraction set to {last_color_idx}: {np.sum(final_coloring == last_color_idx) / final_coloring.size * 100:.8f}%")
np.save(f"equal_parallelograms/{run_id}_{max_size}_fixed", final_coloring)

import os
import csv

def save_fixed_fraction(run_id, stepsize, fraction_fixed, csv_filename='verified_paralellograms.csv'):
    file_exists = os.path.isfile(csv_filename)
    data = {
        'run_id': run_id,
        'n_colors': last_color_idx + 1,
        'stepsize': stepsize,
        'fraction_fixed_to_bonus (%)': round(fraction_fixed * 100, 8)
    }

    with open(csv_filename, mode='a', newline='') as csvfile:
        fieldnames = ['run_id', 'n_colors', 'stepsize', 'fraction_fixed_to_bonus (%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

fraction_fixed = np.sum(final_coloring == last_color_idx) / final_coloring.size
save_fixed_fraction(run_id, max_size, fraction_fixed, csv_filename='EQUAL_PARALLELOGRAMS.csv')

if max(N1, N2) < 500:

    if max(N1, N2) <= 200: 
        suffix = ".pdf"
    else:
        suffix = ".png"

    starts_reshaped = starts.reshape(N1, N2, 2)

    plot_parallelogram_coloring(starts_reshaped, final_coloring, parallelogram[0], parallelogram[1], max_size, f"equal_parallelograms/{run_id}_{max_size}_final" + suffix)