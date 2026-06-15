import argparse
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
parser.add_argument('--eval_gridsize', type=int, default=80)
args = parser.parse_args()

run_id = args.run_id
#run_id = "c04fwjda"
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

def plot_parallelogram_coloring(starts, values, v1, v2, N, filename):

    dv1 = v1.cpu() / N
    dv2 = v2.cpu() / N
    starts = starts.reshape(N*N, 2).cpu()
    values = values.flatten()
    
    cmap = get_cmap()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for p, val in tqdm(zip(starts, values), total=N**2):
        corners = np.stack([
            p,
            p + dv1,
            p + dv1 + dv2,
            p + dv2
        ])
        polygon = Polygon(corners, closed=True, facecolor=cmap(val / last_color_idx), edgecolor='none')
        ax.add_patch(polygon)
    
    # ax.set_aspect('equal')
    # ax.autoscale_view()
    ax.axis('off')

    # add v1 and v2 arrows
    ax.arrow(0, 0, v1[0].item(), v1[1].item(), head_width=0.05, head_length=0.2, fc='black', ec='black')
    ax.arrow(0, 0, v2[0].item(), v2[1].item(), head_width=0.05, head_length=0.2, fc='black', ec='black')

    # add length of the vectors
    ax.text(v1[0].item() / 2, v1[1].item() / 2, f"{torch.norm(v1).item():.2f}", fontsize=12, ha='center', va='center')
    ax.text(v2[0].item() / 2, v2[1].item() / 2, f"{torch.norm(v2).item():.2f}", fontsize=12, ha='center', va='center')


    max_coords = max(v1.max(), v2.max()).item()
    ax.set_xlim(-max_coords, max_coords)
    ax.set_ylim(-max_coords, max_coords)

    plt.savefig(filename)
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
gridsize = args.eval_gridsize
n_colours = run.config['n_colours']
last_color_idx = n_colours - 1
grid_bounds = (3, 3)


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

with torch.no_grad():
    outputs = model(starts.to(device)).argmax(dim=1).cpu()
    grid_coloring = outputs.reshape(gridsize, gridsize).numpy().astype(np.int8)

starts_reshaped = starts.reshape(gridsize, gridsize, 2)
plot_parallelogram_coloring(starts_reshaped, grid_coloring, parallelogram[0], parallelogram[1], gridsize, f"fixed_colorings/{run_id}_{gridsize}_initial.pdf")

# ----------------------
# Conflict Detection
# ----------------------
def init_worker(grid_coloring, mask, num_pixels):
    global global_coloring, global_mask, global_num_pixels
    global_coloring = grid_coloring
    global_mask = mask
    global_num_pixels = num_pixels

def process_center(center):
    i, j = center
    cc = global_coloring[j, i]
    conflicts, used_colors = [], set()

    for di, dj in global_mask:
        ni, nj = (i + di) % global_num_pixels, (j + dj) % global_num_pixels
        # neighbor color
        nc = global_coloring[nj, ni]
        if nc == cc and nc != last_color_idx:
            conflicts.append((ni, nj))
        if nc != last_color_idx:
            used_colors.add(nc)
            # 
    # DO NOT ADD LAST COLOR TO UNUSED, IT DOESN'T MATTER
    unused = [k for k in range(last_color_idx - 1) if k not in used_colors]
    return cc, unused, conflicts

def process_all_centers(grid_coloring, mask):
    num_pixels = grid_coloring.shape[0]
    centers = [(i, j) for i in range(num_pixels) for j in range(num_pixels)]
    with Pool(processes=4, initializer=init_worker, initargs=(grid_coloring, mask, num_pixels)) as pool:
        results = list(tqdm(pool.imap(process_center, centers), total=len(centers)))
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
def fix_coloring(grid_coloring, mask, five_fixable, conflict_fixable, min_vertex_cover):
    fixed_coloring = np.copy(grid_coloring)
    num_pixels = grid_coloring.shape[0]

    for center in five_fixable:
        used_colors = set()
        for di, dj in mask:
            ni, nj = (center[0] + di) % num_pixels, (center[1] + dj) % num_pixels
            mc = grid_coloring[nj, ni]
            if mc != last_color_idx:
                used_colors.add(mc)
        unused = [i for i in range(last_color_idx - 1) if i not in used_colors]
        if unused:
            # choose a random unused color
            fixed_coloring[center[1], center[0]] = unused[0]

    for center in conflict_fixable:
        used_colors = set()
        for di, dj in mask:
            ni, nj = (center[0] + di) % num_pixels, (center[1] + dj) % num_pixels
            mc = fixed_coloring[nj, ni]
            if mc != last_color_idx:
                used_colors.add(mc)
        unused = [i for i in range(last_color_idx - 1) if i not in used_colors]
        if unused:
            fixed_coloring[center[1], center[0]] = unused[0]

    for center in min_vertex_cover:
        fixed_coloring[center[1], center[0]] = last_color_idx

    return fixed_coloring

def fix_coloring_trivial(grid_coloring, mask, five_fixable, conflict_fixable, min_vertex_cover):
    fixed_coloring = np.copy(grid_coloring)
    num_pixels = grid_coloring.shape[0]

    for center in five_fixable:
        fixed_coloring[center[1], center[0]] = last_color_idx 

    for center in conflict_fixable:
        fixed_coloring[center[1], center[0]] = last_color_idx

    for center in min_vertex_cover:
        fixed_coloring[center[1], center[0]] = last_color_idx

    return fixed_coloring

# ----------------------
# Iterative Fixing
# ----------------------




centers, results = process_all_centers(grid_coloring, base_mask)
stats = analyze_conflicts(results, centers, grid_coloring)

count = 0

total_bad = len(stats["conflict_non_fixable"]) + len(stats["conflict_fixable"]) + len(stats["five_non_fixable"]) + len(stats["five_fixable"])

print(f"Initial stats: \n Non-fixable conflicts: {len(stats['conflict_non_fixable'])} ({len(stats['conflict_non_fixable']) / (gridsize**2) * 100:.8f}%)\n Fixable conflicts: {len(stats['conflict_fixable'])} ({len(stats['conflict_fixable']) / (gridsize**2) * 100:.8f}%)\n Non-fixable five: {len(stats['five_non_fixable'])} ({len(stats['five_non_fixable']) / (gridsize**2) * 100:.8f}%)\n Fixable five: {len(stats['five_fixable'])} ({len(stats['five_fixable']) / (gridsize**2) * 100:.8f}%)\n Total bad: {total_bad} ({total_bad / (gridsize**2) * 100:.8f}%)\n")

fixed_coloring = fix_coloring(grid_coloring, base_mask, stats["five_fixable"], stats["conflict_fixable"], stats["min_vertex_cover"])

if gridsize < 500:

    initial_fix = fix_coloring_trivial(grid_coloring, base_mask, stats["five_fixable"], stats["conflict_fixable"], stats["min_vertex_cover"])
    if gridsize <= 300: 
        suffix = ".pdf"
    else:
        suffix = ".png"

    starts_reshaped = starts.reshape(gridsize, gridsize, 2)
    plot_parallelogram_coloring(starts_reshaped, initial_fix, parallelogram[0], parallelogram[1], gridsize, f"fixed_colorings/{run_id}_{gridsize}_trivial" + suffix)




next_iteration = fixed_coloring


while len(stats["conflict_fixable"]) + len(stats["conflict_non_fixable"]) + len(stats["five_fixable"]) > 0:


    count += 1
    next_iteration = fix_coloring(next_iteration, base_mask, stats["five_fixable"], stats["conflict_fixable"], stats["min_vertex_cover"])
    centers, results = process_all_centers(next_iteration, base_mask)
    stats = analyze_conflicts(results, centers, next_iteration)

    total_bad = len(stats["conflict_non_fixable"]) + len(stats["conflict_fixable"]) + len(stats["five_non_fixable"]) + len(stats["five_fixable"])

    print(f"Iteration {count}: \n Non-fixable conflicts: {len(stats['conflict_non_fixable'])} ({len(stats['conflict_non_fixable']) / (gridsize**2) * 100:.8f}%)\n Fixable conflicts: {len(stats['conflict_fixable'])} ({len(stats['conflict_fixable']) / (gridsize**2) * 100:.8f}%)\n Non-fixable five: {len(stats['five_non_fixable'])} ({len(stats['five_non_fixable']) / (gridsize**2) * 100:.8f}%)\n Fixable five: {len(stats['five_fixable'])} ({len(stats['five_fixable']) / (gridsize**2) * 100:.8f}%)\n Total bad: {total_bad} ({total_bad / (gridsize**2) * 100:.8f}%)\n")

    if count > 10:
        for group in ["five_non_fixable", "conflict_non_fixable", "five_fixable", "conflict_fixable"]:
            for center in stats[group]:
                next_iteration[center[1], center[0]] = last_color_idx
        break

print(f"Converged after {count} iterations.")
final_coloring = next_iteration
print(f"Fraction set to {last_color_idx}: {np.sum(final_coloring == last_color_idx) / final_coloring.size * 100:.8f}%")
np.save(f"fixed_colorings/{run_id}_{gridsize}_fixed", final_coloring)

import os
import csv

def save_fixed_fraction(run_id, eval_gridsize, fraction_fixed, csv_filename='verified_paralellograms.csv'):
    file_exists = os.path.isfile(csv_filename)
    data = {
        'run_id': run_id,
        'n_colors': last_color_idx + 1,
        'eval_gridsize': eval_gridsize,
        'fraction_fixed_to_bonus (%)': round(fraction_fixed * 100, 8)
    }

    with open(csv_filename, mode='a', newline='') as csvfile:
        fieldnames = ['run_id', 'n_colors', 'eval_gridsize', 'fraction_fixed_to_bonus (%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

fraction_fixed = np.sum(final_coloring == last_color_idx) / final_coloring.size
save_fixed_fraction(run_id, gridsize, fraction_fixed, csv_filename='PARALLELOGRAMS_FINAL.csv')

if gridsize < 500:

    if gridsize <= 300: 
        suffix = ".pdf"
    else:
        suffix = ".png"

    starts_reshaped = starts.reshape(gridsize, gridsize, 2)

    plot_parallelogram_coloring(starts_reshaped, final_coloring, parallelogram[0], parallelogram[1], gridsize, f"fixed_colorings/{run_id}_{gridsize}_final" + suffix)