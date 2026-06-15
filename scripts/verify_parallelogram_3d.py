import argparse
import math
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import wandb
import networkx as nx
import os
import csv

# ensure parent modules are importable
sys.path.append("..")
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from models import ResMLP
from utilities import GeneralUtility

# ----------------------
# Argument Parsing
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=str, required=True)
parser.add_argument('--max_size', type=float, default=0.01)
parser.add_argument('--radius', type=float, default=1.0)
args = parser.parse_args()
run_id = args.run_id
max_size = args.max_size
radius = args.radius

# ----------------------
# Load W&B Run
# ----------------------
api = wandb.Api()
run = api.run(f"ais2t/3DHadwigerNelson/{run_id}")

# download trained checkpoint
def find_model_file(run):

    for file in run.files():
        if "trained_model" in file.name:
            
            return file.name
checkpoint_file = find_model_file(run)
ckpt = run.file(checkpoint_file).download(root=f"./models/{run_id}", replace=True)

# ----------------------
# Parallelpipetral Mask Computation
# ----------------------
def compute_base_mask_vec_3d(parallelepiped, max_size, radius=1.0, device='cpu'):
    # parallelepiped: tensor of shape (3,3): rows = basis vectors v1,v2,v3
    v1, v2, v3 = parallelepiped[0].to(device), parallelepiped[1].to(device), parallelepiped[2].to(device)
    # subdivisions
    L1, L2, L3 = v1.norm().item(), v2.norm().item(), v3.norm().item()
    N1 = max(1, math.ceil(L1 / max_size))
    N2 = max(1, math.ceil(L2 / max_size))
    N3 = max(1, math.ceil(L3 / max_size))

    print(f"Grid: N1={N1}, N2={N2}, N3={N3} \n L1={L1:.4f}, L2={L2:.4f}, L3={L3:.4f} \n Total cells: {N1 * N2 * N3}")

    # fundamental starts
    lin1 = torch.linspace(0, 1, N1+1, device=device)[:-1]
    lin2 = torch.linspace(0, 1, N2+1, device=device)[:-1]
    lin3 = torch.linspace(0, 1, N3+1, device=device)[:-1]
    a, b, c = torch.meshgrid(lin1, lin2, lin3, indexing='ij')  # shape N1×N2×N3
    starts = (a.reshape(-1,1)*v1 + b.reshape(-1,1)*v2 + c.reshape(-1,1)*v3)  # [N1*N2*N3,3]

    # tile offsets in 3x3x3
    offs = torch.stack(
        torch.meshgrid(
            torch.tensor([-1,0,1], device=device),
            torch.tensor([-1,0,1], device=device),
            torch.tensor([-1,0,1], device=device),
            indexing='ij'
        ), -1
    ).view(-1,3)  # [27,3]
    tile_shifts = offs[:,0:1]*v1 + offs[:,1:2]*v2 + offs[:,2:3]*v3  # [27,3]

    # all shifted starts
    shifted = (starts.unsqueeze(0) + tile_shifts.unsqueeze(1)).view(-1,3)  # [27*Ncells,3]

    # corner offsets (8 corners)
    dv1, dv2, dv3 = v1 / N1, v2 / N2, v3 / N3
    corner_offs = torch.stack([
        torch.zeros(3, device=device),
        dv1, dv2, dv3,
        dv1 + dv2, dv1 + dv3, dv2 + dv3,
        dv1 + dv2 + dv3
    ], dim=0)  # [8,3]

    # compute corner positions
    corners = shifted.unsqueeze(1) + corner_offs.unsqueeze(0)  # [27*Ncells,8,3]
    centers = corner_offs  # [8,3]

    # squared distances
    diffs = corners.unsqueeze(2) - centers.view(1,1,8,3)
    d2 = (diffs**2).sum(dim=-1)  # [27*Ncells,8,8]

    inside      = d2 <= radius*radius
    any_inside  = inside.any(dim=1)
    all_inside  = inside.all(dim=1)
    partial     = (any_inside & (~all_inside)).any(dim=1)  # [27*Ncells]

    idxs = torch.nonzero(partial, as_tuple=False).squeeze(1)
    total_cells = N1 * N2 * N3
    cell_idx = idxs % total_cells

    # decompose to i,j,k
    i = cell_idx // (N2 * N3)
    rem = cell_idx % (N2 * N3)
    j = rem // N3
    k = rem % N3

    base_mask = torch.stack([i, j, k], dim=1).tolist()

    print(f"Mask size: {len(base_mask)}")
    return starts.cpu(), base_mask, (N1, N2, N3)

# ----------------------
# Conflict Detection (3D)
# ----------------------
# global placeholders for multiprocess

def init_worker(grid_coloring, mask):
    global global_coloring, global_mask, global_n1, global_n2, global_n3
    global_coloring = grid_coloring
    global_mask     = mask
    global_n1, global_n2, global_n3 = grid_coloring.shape


def process_center(center):
    i, j, k = center
    cc = int(global_coloring[i, j, k].item())

    conflicts   = []
    used_colors = set()
    for di, dj, dk in global_mask:
        ni = (i + di) % global_n1
        nj = (j + dj) % global_n2
        nk = (k + dk) % global_n3
        nc = int(global_coloring[ni, nj, nk].item())
        if nc == cc and nc != last_color_idx:
            conflicts.append((ni, nj, nk))
        if nc != last_color_idx:
            used_colors.add(nc)

    unused = [k for k in range(last_color_idx) if k not in used_colors]
    return cc, unused, conflicts


def process_all_centers(grid_coloring, mask, num_workers=32):
    N1, N2, N3 = grid_coloring.shape
    centers = [(i, j, k) for i in range(N1) for j in range(N2) for k in range(N3)]
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
# Analyze & Fix
# ----------------------
def analyze_conflicts(results, centers, grid_coloring):
    five_non_fixable = []
    five_fixable = []
    conflict_non_fixable = {}
    conflict_fixable = []

    for center, (cc, unused_colors, conflicts) in zip(centers, results):
        if cc == last_color_idx:
            if unused_colors:
                five_fixable.append(center)
            else:
                five_non_fixable.append(center)
        else:
            if conflicts:
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

    total = np.prod(grid_coloring.shape)
    percent_remaining = (len(five_non_fixable) + len(min_vertex_cover)) / total * 100
    return {
        "five_non_fixable": five_non_fixable,
        "five_fixable": five_fixable,
        "conflict_non_fixable": conflict_non_fixable,
        "conflict_fixable": conflict_fixable,
        "min_vertex_cover": min_vertex_cover,
        "percent_remaining_conflicts": percent_remaining,
    }


def fix_coloring(grid_coloring, mask, five_fixable, conflict_fixable, min_vertex_cover):
    if isinstance(grid_coloring, torch.Tensor):
        grid_np = grid_coloring.cpu().numpy()
    else:
        grid_np = np.array(grid_coloring, copy=True)
    fixed = grid_np.copy()
    N1, N2, N3 = grid_np.shape

    # fix five-fixable
    for (i, j, k) in five_fixable:
        used_colors = set()
        for di, dj, dk in mask:
            ni = (i + di) % N1
            nj = (j + dj) % N2
            nk = (k + dk) % N3
            mc = grid_np[ni, nj, nk]
            if mc != last_color_idx:
                used_colors.add(int(mc))
        unused = [c for c in range(last_color_idx) if c not in used_colors]
        if unused:
            fixed[i, j, k] = unused[0]

    # fix conflict-fixable
    for (i, j, k) in conflict_fixable:
        used_colors = set()
        for di, dj, dk in mask:
            ni = (i + di) % N1
            nj = (j + dj) % N2
            nk = (k + dk) % N3
            mc = fixed[ni, nj, nk]
            if mc != last_color_idx:
                used_colors.add(int(mc))
        unused = [c for c in range(last_color_idx) if c not in used_colors]
        if unused:
            fixed[i, j, k] = unused[0]

    # blank min-vertex-cover
    for (i, j, k) in min_vertex_cover:
        fixed[i, j, k] = last_color_idx
    return fixed

# ----------------------
# Main Pipeline
# ----------------------
if __name__ == '__main__':
    # load model
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device="cpu"
    input_dim = run.config['dim']  # should be 3 for 3D
    n_colours = run.config['n_colours']
    last_color_idx = n_colours - 1
    model = ResMLP(input_dim=input_dim, output_dim=n_colours, device=device, **run.config['model'])
    model.load_state_dict(torch.load(ckpt.name, map_location=device))
    model = model.to(device).eval()

    # get parallelepiped
    if run.config['training'].get('trainable_parallelogram', False):
        ps = []
        for i in range(3):
            row = []
            for j in range(3):
                key = f'parallelogram_{i}_{j}'
                row.append(run.summary[key])
            ps.append(row)
        parallelepiped = torch.tensor(ps, device=device)
    else:
        parallelepiped = torch.tensor(run.config['training']['parallelogram'], device=device)

    model = GeneralUtility.prepend_parallelogram_transformation(model, spanning_vectors=parallelepiped)

    # compute base mask and starts
    starts, base_mask, (N1, N2, N3) = compute_base_mask_vec_3d(parallelepiped, max_size, radius, device=device)

    # model evaluation
    with torch.no_grad():
        outputs = model(starts.to(device)).argmax(dim=1).cpu()
        grid_coloring = outputs.reshape(N1, N2, N3).numpy().astype(np.int8)

    # conflict detection
    centers, results = process_all_centers(torch.tensor(grid_coloring), base_mask)
    stats = analyze_conflicts(results, centers, grid_coloring)

    # initial stats
    total_cells = N1 * N2 * N3
    bad = (len(stats['five_non_fixable']) + len(stats['five_fixable']) +
           len(stats['conflict_non_fixable']) + len(stats['conflict_fixable']))
    print(f"Initial bad cells: {bad}/{total_cells} ({bad/total_cells*100:.4f}%)")

    # first fix
    fixed_coloring = fix_coloring(grid_coloring, base_mask,
                                  stats['five_fixable'], stats['conflict_fixable'],
                                  stats['min_vertex_cover'])

    # iterative fixing
    count = 0
    next_col = fixed_coloring
    while len(stats['five_fixable']) + len(stats['conflict_fixable']) > 0 and count < 10:
        count += 1
        next_col = fix_coloring(next_col, base_mask,
                                stats['five_fixable'], stats['conflict_fixable'],
                                stats['min_vertex_cover'])
        centers, results = process_all_centers(torch.tensor(next_col), base_mask)
        stats = analyze_conflicts(results, centers, next_col)
        bad = (len(stats['five_non_fixable']) + len(stats['five_fixable']) +
               len(stats['conflict_non_fixable']) + len(stats['conflict_fixable']))
        print(f"Iteration {count} bad: {bad}/{total_cells} ({bad/total_cells*100:.4f}%)")
        if count == 10:
            # force-blank remaining
            for group in ['five_non_fixable', 'conflict_non_fixable', 'five_fixable', 'conflict_fixable']:
                for center in stats[group]:
                    next_col[center] = last_color_idx
            break

    final_coloring = next_col
    # save volume
    os.makedirs('outputs', exist_ok=True)
    np.save(f"outputs/{run_id}_{max_size}_fixed3d.npy", final_coloring)

    # record stats
    fraction_fixed = np.sum(final_coloring == last_color_idx) / final_coloring.size
    csv_file = 'EQUAL_PARALLELEPIPEDS.csv'
    exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['run_id','n_colours','stepsize','fraction_fixed'])
        if not exists:
            writer.writeheader()
        writer.writerow({
            'run_id': run_id,
            'n_colours': n_colours,
            'stepsize': max_size,
            'fraction_fixed': round(fraction_fixed*100,4)
        })
    print(f"Done: {fraction_fixed*100:.4f}% set to blank")
