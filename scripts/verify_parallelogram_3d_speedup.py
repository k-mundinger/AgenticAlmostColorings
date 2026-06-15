import argparse
import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
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
# Analyze & Fix
# ----------------------


def detect_conflicts_vec(grid, mask, last_color_idx):
    """
    grid:             torch.Tensor [N1,N2,N3], dtype long/int
    mask:             list of (di,dj,dk) neighbor offsets
    last_color_idx:   int index of the "blank" color
    returns: dict with keys
      five_fixable, five_non_fixable,
      conflict_fixable, conflict_non_fixable,
      min_vertex_cover
    """
    device = grid.device
    N1, N2, N3 = grid.shape
    M = len(mask)

    # 1) roll the grid for each neighbor offset → [M,N1,N2,N3]
    rolls = torch.stack([
        grid.roll(shifts=(di, dj, dk), dims=(0,1,2))
        for di, dj, dk in mask
    ], dim=0)

    # 2) detect conflicts and occupancy
    same_color   = (rolls == grid.unsqueeze(0))          # [M,N1,N2,N3]
    non_blank    = (rolls != last_color_idx)             # [M,N1,N2,N3]
    has_conflict = (same_color & non_blank).any(dim=0)   # [N1,N2,N3]

    # 3) compute used vs unused real colors
    num_classes = last_color_idx + 1
    oh = F.one_hot(rolls.long(), num_classes=num_classes)   # [M,N1,N2,N3,C+1]
    used = oh[..., :last_color_idx].any(dim=0)               # [N1,N2,N3,C]
    unused_count = (~used).sum(dim=-1)                       # [N1,N2,N3]

    # 4) masks for each category
    is_blank            = (grid == last_color_idx)
    non_blank_cells     = ~is_blank
    five_fixable_mask   = is_blank & (unused_count > 0)
    five_nonfix_mask    = is_blank & (unused_count == 0)
    conflict_fixable_mask = non_blank_cells & has_conflict & (unused_count > 0)
    conflict_nonfix_mask  = non_blank_cells & has_conflict & (unused_count == 0)

    # 5) helper to extract (i,j,k) positions from a boolean mask
    idxs = torch.arange(N1*N2*N3, device=device).view(N1,N2,N3)
    def extract(mask_tensor):
        flat = idxs[mask_tensor]
        i = flat // (N2*N3)
        rem = flat % (N2*N3)
        j = rem // N3
        k = rem % N3
        return list(zip(i.cpu().tolist(), j.cpu().tolist(), k.cpu().tolist()))

    five_fixable        = extract(five_fixable_mask)
    five_non_fixable    = extract(five_nonfix_mask)
    conflict_fixable    = extract(conflict_fixable_mask)
    conflict_non_fixable = extract(conflict_nonfix_mask)

    # 6) build conflict graph for non-fixable conflicts
    G = nx.Graph()
    G.add_nodes_from(conflict_non_fixable)
    conf_set = set(conflict_non_fixable)
    for (i, j, k) in conflict_non_fixable:
        for di, dj, dk in mask:
            nei = ((i + di) % N1, (j + dj) % N2, (k + dk) % N3)
            if nei in conf_set:
                G.add_edge((i, j, k), nei)

    # 7) approximate min vertex cover
    mvc = nx.algorithms.approximation.min_weighted_vertex_cover(G)

    return {
        "five_fixable": five_fixable,
        "five_non_fixable": five_non_fixable,
        "conflict_fixable": conflict_fixable,
        "conflict_non_fixable": conflict_non_fixable,
        "min_vertex_cover": list(mvc),
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
    stats = detect_conflicts_vec(
    torch.tensor(grid_coloring, device=device),
    base_mask,
    last_color_idx
)

    print(f"Initial bad: "
        f"{len(stats['five_non_fixable']) + len(stats['conflict_non_fixable'])} "
        f"/ {N1*N2*N3}")

    # apply first fix
    next_col = fix_coloring(
        grid_coloring,
        base_mask,
        stats['five_fixable'],
        stats['conflict_fixable'],
        stats['min_vertex_cover']
    )

    # iterate until no more fixables (or max 10 rounds)
    for it in range(1,11):
        stats = detect_conflicts_vec(
            torch.tensor(next_col, device=device),
            base_mask,
            last_color_idx
        )
        total_bad = (len(stats['five_non_fixable']) +
                    len(stats['conflict_non_fixable']))
        print(f"Iter {it} bad: {total_bad}/{N1*N2*N3}")
        if not stats['five_fixable'] and not stats['conflict_fixable']:
            break
        next_col = fix_coloring(
            next_col,
            base_mask,
            stats['five_fixable'],
            stats['conflict_fixable'],
            stats['min_vertex_cover']
        )

    # final result in `next_col`
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
