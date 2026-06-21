import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

_default_mplconfig = Path(os.getenv("TMPDIR", "/tmp")) / "aac_mplconfig"
_default_mplconfig.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_default_mplconfig))

from models import ResMLP
from utilities import GeneralUtility


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Heuristic almost-coloring verifier/repair path. This avoids the "
            "large component MILP and only emits a coloring after a full "
            "real-color conflict validation."
        )
    )
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--eval_gridsize", type=int, default=512)
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--mask_cache_dir", default=None)
    parser.add_argument("--neighbor_backend", choices=("fft", "conv"), default="fft")
    parser.add_argument("--passes", type=int, default=6)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate_fractions",
        default="1.0,0.5,0.25,0.125,0.0625",
        help="Comma-separated acceptance fractions cycled over heuristic passes.",
    )
    parser.add_argument(
        "--fill_strategy",
        choices=("isolated", "mis"),
        default="mis",
        help=(
            "isolated keeps only proposals with no proposed neighbor; mis builds "
            "a candidate graph and keeps a randomized maximal independent set."
        ),
    )
    parser.add_argument(
        "--mis_edge_chunk_size",
        type=int,
        default=32,
        help="Number of unit-distance offsets processed per chunk when building MIS edges.",
    )
    parser.add_argument(
        "--mis_max_rounds",
        type=int,
        default=64,
        help="Maximum Luby-style rounds for each candidate MIS solve.",
    )
    parser.add_argument(
        "--augment_rounds",
        type=int,
        default=0,
        help=(
            "Extra local-search rounds after constructive fill. Each round tries "
            "bonus->real moves whose single active blocker can be recolored."
        ),
    )
    parser.add_argument(
        "--augment_max_proposals_per_color",
        type=int,
        default=200000,
        help="Cap accepted proposals considered per target color in each augment round.",
    )
    parser.add_argument(
        "--kick_rounds",
        type=int,
        default=0,
        help=(
            "Extra escape rounds after augmentation. A kick colors a bonus cell "
            "by turning its single active blocker back to bonus, then reruns MIS."
        ),
    )
    parser.add_argument(
        "--kick_max_proposals_per_color",
        type=int,
        default=50000,
        help="Cap kick proposals per target color in each kick round.",
    )
    parser.add_argument(
        "--color_order",
        choices=("fixed", "shuffle"),
        default="shuffle",
        help="Order for trying real colors in each pass.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--no_plot", action="store_true")
    return parser.parse_args()


args = parse_args()
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


def parse_candidate_fractions(value):
    fractions = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        fraction = float(item)
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"candidate fraction must be in (0, 1], got {fraction}")
        fractions.append(fraction)
    if not fractions:
        raise ValueError("At least one candidate fraction is required.")
    return fractions


def select_device():
    if args.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")
    return args.device


device = select_device()
candidate_fractions = parse_candidate_fractions(args.candidate_fractions)


def load_model(config, checkpoint_path):
    model = ResMLP(
        input_dim=config["dim"],
        output_dim=config["n_colours"],
        device=device,
        **config["model"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()

    parallelogram = torch.tensor(
        config["training"]["parallelogram"],
        device=device,
        dtype=torch.float32,
    )
    model = GeneralUtility.prepend_parallelogram_transformation(
        model,
        spanning_vectors=parallelogram,
    )
    return model, parallelogram


def resolve_mask_cache_dir():
    if args.mask_cache_dir == "none":
        return None
    if args.mask_cache_dir is not None:
        return Path(args.mask_cache_dir)
    return output_dir.parent / "mask_cache"


def mask_cache_path(parallelogram, n, radius):
    cache_dir = resolve_mask_cache_dir()
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_data = [
        np.asarray(parallelogram.detach().cpu().numpy(), dtype=np.float32).tobytes(),
        str(n).encode("ascii"),
        repr(float(radius)).encode("ascii"),
    ]
    digest = hashlib.sha256(b"|".join(key_data)).hexdigest()[:20]
    return cache_dir / f"base_mask_{n}_{digest}.npy"


def compute_base_mask_vec(parallelogram, n, radius=1.0):
    v1, v2 = parallelogram[0], parallelogram[1]
    lin = torch.linspace(0, 1, n + 1, device=device, dtype=v1.dtype)[:-1]
    a, b = torch.meshgrid(lin, lin, indexing="ij")
    starts_grid = a.unsqueeze(-1) * v1 + b.unsqueeze(-1) * v2

    cache_path = mask_cache_path(parallelogram, n, radius)
    if cache_path is not None and cache_path.exists():
        base_mask = np.load(cache_path).astype(np.int64, copy=False).tolist()
        print(f"Loaded cached base mask from {cache_path}", flush=True)
        return starts_grid.reshape(-1, 2).cpu(), base_mask

    dv1, dv2 = v1 / n, v2 / n
    corner_offsets = torch.stack(
        [
            torch.zeros(2, device=device, dtype=v1.dtype),
            dv1,
            dv1 + dv2,
            dv2,
        ],
        dim=0,
    )
    tile_offsets = torch.tensor(
        [[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1]],
        device=device,
        dtype=v1.dtype,
    )

    base_mask = []
    radius_sq = radius * radius
    for tile_offset in tile_offsets:
        tile_partial = torch.zeros((n, n), dtype=torch.bool, device=device)
        shift = tile_offset[0] * v1 + tile_offset[1] * v2
        corners = starts_grid.unsqueeze(0) + shift + corner_offsets[:, None, None, :]
        for center in corner_offsets:
            d2 = ((corners - center.view(1, 1, 1, 2)) ** 2).sum(dim=-1)
            inside = d2 <= radius_sq
            tile_partial |= inside.any(dim=0) & (~inside.all(dim=0))
        base_mask.extend(torch.nonzero(tile_partial, as_tuple=False).tolist())

    if cache_path is not None:
        np.save(cache_path, np.asarray(base_mask, dtype=np.int32))
        print(f"Saved base mask cache to {cache_path}", flush=True)

    return starts_grid.reshape(-1, 2).cpu(), base_mask


def one_hot_encode(grid, colors):
    return F.one_hot(grid.long(), colors).permute(2, 0, 1).float()


def build_fft_mask_kernel(mask_offsets, height, width):
    kernel = torch.zeros((height, width), device=device)
    for di, dj in mask_offsets:
        kernel[(-int(di)) % height, (-int(dj)) % width] = 1.0
    return torch.fft.rfft2(kernel)


def build_mask_kernel(mask_offsets):
    dis, djs = mask_offsets[:, 0], mask_offsets[:, 1]
    di_min, di_max = int(dis.min()), int(dis.max())
    dj_min, dj_max = int(djs.min()), int(djs.max())
    kh, kw = di_max - di_min + 1, dj_max - dj_min + 1

    kernel = torch.zeros((1, 1, kh, kw), device=device)
    for di, dj in mask_offsets:
        kernel[0, 0, int(di - di_min), int(dj - dj_min)] = 1.0
    pad = (-dj_min, dj_max, -di_min, di_max)
    return kernel, pad


def count_same_color_neighbors(onehot, mask_kernel, pad):
    if pad is None:
        height, width = onehot.shape[-2:]
        counts = torch.fft.irfft2(
            torch.fft.rfft2(onehot) * mask_kernel,
            s=(height, width),
        )
        return counts.round_().clamp_min_(0.0)

    x = F.pad(onehot.unsqueeze(0), pad=pad, mode="circular")
    colors = onehot.shape[0]
    weight = mask_kernel.repeat(colors, 1, 1, 1)
    return F.conv2d(x, weight=weight, groups=colors).squeeze(0)


def count_mask_neighbors(mask, mask_kernel, pad):
    if pad is None:
        height, width = mask.shape
        counts = torch.fft.irfft2(
            torch.fft.rfft2(mask.float()) * mask_kernel,
            s=(height, width),
        )
        return counts.round_().clamp_min_(0.0)

    x = F.pad(mask.float().view(1, 1, *mask.shape), pad=pad, mode="circular")
    return F.conv2d(x, weight=mask_kernel).view(mask.shape)


def compute_used_colors(grid, colors, mask_kernel, pad):
    onehot = one_hot_encode(grid, colors)
    return count_same_color_neighbors(onehot, mask_kernel, pad) > 0


def real_conflict_mask(grid, colors, mask_kernel, pad):
    onehot = one_hot_encode(grid, colors)
    counts = count_same_color_neighbors(onehot, mask_kernel, pad)
    height, width = grid.shape
    flat_counts = counts.view(colors, -1)
    flat_grid = grid.view(-1).long()
    flat_idx = torch.arange(height * width, device=grid.device)
    selected = flat_counts[flat_grid, flat_idx].view(height, width)
    return (grid < colors - 1) & (selected > 0)


def apply_initial_greedy_repairs(grid, colors, mask_kernel, pad):
    bonus = colors - 1

    phase_start = perf_counter()
    used = compute_used_colors(grid, colors, mask_kernel, pad)
    bonus_mask = grid == bonus
    no_real_neighbors = bonus_mask & (~used[:bonus].any(dim=0))
    for color in range(bonus):
        grid[no_real_neighbors & (~used[color])] = color
    print(f"Phase 1 bonus greedy repair took {perf_counter() - phase_start:.2f}s", flush=True)

    phase_start = perf_counter()
    conflicts = real_conflict_mask(grid, colors, mask_kernel, pad)
    used = compute_used_colors(grid, colors, mask_kernel, pad)
    no_real_neighbors = conflicts & (~used[:bonus].any(dim=0))
    for color in range(bonus):
        grid[no_real_neighbors & (~used[color])] = color
    print(f"Phase 2 conflict greedy repair took {perf_counter() - phase_start:.2f}s", flush=True)

    phase_start = perf_counter()
    conflicts = real_conflict_mask(grid, colors, mask_kernel, pad)
    print(
        f"Phase 3 conflict detection found {int(conflicts.sum().item())} cells "
        f"in {perf_counter() - phase_start:.2f}s",
        flush=True,
    )
    return grid, conflicts


def bonus_fraction(grid, colors):
    return float((grid == colors - 1).sum().item()) / float(grid.numel())


def color_order_for_pass(real_colors, restart, pass_index):
    order = list(range(real_colors))
    if args.color_order == "shuffle":
        rng = np.random.default_rng(args.seed + restart * 1009 + pass_index * 9176)
        rng.shuffle(order)
    return order


def unique_offsets(mask_offsets):
    return np.asarray(
        list(dict.fromkeys((int(di), int(dj)) for di, dj in mask_offsets)),
        dtype=np.int64,
    )


def build_candidate_edges(candidate_mask, mask_offsets):
    coords = torch.nonzero(candidate_mask, as_tuple=False)
    num_candidates = int(coords.shape[0])
    if num_candidates <= 1:
        return coords, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    height, width = candidate_mask.shape
    edge_chunk_size = max(1, int(args.mis_edge_chunk_size))
    candidate_index = torch.full(
        (height, width),
        -1,
        dtype=torch.int32,
        device=candidate_mask.device,
    )
    candidate_index[coords[:, 0], coords[:, 1]] = torch.arange(
        num_candidates,
        dtype=torch.int32,
        device=candidate_mask.device,
    )

    offsets = torch.as_tensor(
        unique_offsets(mask_offsets),
        dtype=torch.long,
        device=candidate_mask.device,
    )
    ci = coords[:, 0]
    cj = coords[:, 1]
    source_indices = torch.arange(
        num_candidates,
        dtype=torch.int32,
        device=candidate_mask.device,
    ).unsqueeze(1)

    left_edges = []
    right_edges = []
    for chunk_start in range(0, int(offsets.shape[0]), edge_chunk_size):
        offset_chunk = offsets[chunk_start:chunk_start + edge_chunk_size]
        ni = (ci.unsqueeze(1) + offset_chunk[:, 0].unsqueeze(0)) % height
        nj = (cj.unsqueeze(1) + offset_chunk[:, 1].unsqueeze(0)) % width
        neighbor_indices = candidate_index[ni, nj]
        edge_mask = neighbor_indices > source_indices
        if not edge_mask.any():
            continue

        left_idx, offset_idx = torch.nonzero(edge_mask, as_tuple=True)
        right_idx = neighbor_indices[left_idx, offset_idx]
        left_edges.append(left_idx.detach().cpu().numpy().astype(np.int32, copy=False))
        right_edges.append(right_idx.detach().cpu().numpy().astype(np.int32, copy=False))

    if not left_edges:
        return coords, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    return coords, np.concatenate(left_edges), np.concatenate(right_edges)


def randomized_maximal_independent_set(num_nodes, left_edges, right_edges, seed):
    if num_nodes == 0:
        return np.zeros(0, dtype=np.bool_)
    if left_edges.size == 0:
        return np.ones(num_nodes, dtype=np.bool_)

    rng = np.random.default_rng(seed)
    active = np.ones(num_nodes, dtype=np.bool_)
    selected = np.zeros(num_nodes, dtype=np.bool_)

    for _ in range(max(1, int(args.mis_max_rounds))):
        if not active.any():
            break

        active_edges = active[left_edges] & active[right_edges]
        if not active_edges.any():
            selected[active] = True
            break

        priorities = rng.random(num_nodes, dtype=np.float32)
        priorities[~active] = -1.0
        max_neighbor_priority = np.full(num_nodes, -1.0, dtype=np.float32)

        left = left_edges[active_edges]
        right = right_edges[active_edges]
        np.maximum.at(max_neighbor_priority, left, priorities[right])
        np.maximum.at(max_neighbor_priority, right, priorities[left])

        chosen = active & (priorities > max_neighbor_priority)
        if not chosen.any():
            chosen[int(np.flatnonzero(active)[0])] = True

        selected |= chosen
        touched_edges = active_edges & (chosen[left_edges] | chosen[right_edges])
        blocked = chosen.copy()
        if touched_edges.any():
            blocked[left_edges[touched_edges]] = True
            blocked[right_edges[touched_edges]] = True
        active[blocked] = False

    if active.any():
        # Fallback keeps correctness if max_rounds is hit: add only isolated
        # remaining active vertices.
        active_edges = active[left_edges] & active[right_edges]
        if active_edges.any():
            incident = np.zeros(num_nodes, dtype=np.bool_)
            incident[left_edges[active_edges]] = True
            incident[right_edges[active_edges]] = True
            selected[active & (~incident)] = True
        else:
            selected[active] = True

    return selected


def fill_active_cells_isolated(base_grid, active_mask, colors, mask_kernel, pad, restart):
    bonus = colors - 1
    grid = base_grid.clone()
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + restart * 1000003)

    previous_bonus = int((active_mask & (grid == bonus)).sum().item())
    for pass_index in range(args.passes):
        fraction = candidate_fractions[pass_index % len(candidate_fractions)]
        pass_assigned = 0
        pass_reverted = 0

        for color in color_order_for_pass(bonus, restart, pass_index):
            active_bonus = active_mask & (grid == bonus)
            if not active_bonus.any():
                break

            used = compute_used_colors(grid, colors, mask_kernel, pad)
            candidates = active_bonus & (~used[color])
            if fraction < 1.0 and candidates.any():
                candidates &= torch.rand(
                    candidates.shape,
                    device=device,
                    generator=generator,
                ) < fraction

            proposed = int(candidates.sum().item())
            if proposed == 0:
                continue

            grid[candidates] = color
            used_after = compute_used_colors(grid, colors, mask_kernel, pad)
            revert = candidates & used_after[color]
            reverted = int(revert.sum().item())
            if reverted:
                grid[revert] = bonus

            pass_assigned += proposed - reverted
            pass_reverted += reverted

        current_bonus = int((active_mask & (grid == bonus)).sum().item())
        print(
            f"Restart {restart} pass {pass_index}: "
            f"fraction={fraction:.4g}, accepted={pass_assigned}, "
            f"reverted={pass_reverted}, active_bonus={current_bonus}",
            flush=True,
        )
        if current_bonus == previous_bonus and pass_assigned == 0:
            break
        previous_bonus = current_bonus

    return grid


def fill_active_cells_mis(base_grid, active_mask, colors, mask_kernel, pad, mask_offsets, restart):
    bonus = colors - 1
    grid = base_grid.clone()
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + restart * 1000003)

    previous_bonus = int((active_mask & (grid == bonus)).sum().item())
    for pass_index in range(args.passes):
        fraction = candidate_fractions[pass_index % len(candidate_fractions)]
        pass_assigned = 0
        pass_edges = 0
        pass_candidates = 0

        for color in color_order_for_pass(bonus, restart, pass_index):
            active_bonus = active_mask & (grid == bonus)
            if not active_bonus.any():
                break

            occupied_color = grid == color
            forbidden = count_mask_neighbors(occupied_color, mask_kernel, pad) > 0
            candidates = active_bonus & (~forbidden)
            if fraction < 1.0 and candidates.any():
                candidates &= torch.rand(
                    candidates.shape,
                    device=device,
                    generator=generator,
                ) < fraction

            candidate_count = int(candidates.sum().item())
            if candidate_count == 0:
                continue

            edge_start = perf_counter()
            coords, left_edges, right_edges = build_candidate_edges(candidates, mask_offsets)
            selected = randomized_maximal_independent_set(
                int(coords.shape[0]),
                left_edges,
                right_edges,
                args.seed + restart * 1000003 + pass_index * 9176 + color * 101,
            )
            selected_count = int(selected.sum())
            if selected_count:
                selected_indices = torch.as_tensor(
                    np.flatnonzero(selected),
                    dtype=torch.long,
                    device=coords.device,
                )
                selected_coords = coords[selected_indices]
                grid[selected_coords[:, 0], selected_coords[:, 1]] = color

            pass_assigned += selected_count
            pass_edges += int(left_edges.size)
            pass_candidates += candidate_count
            print(
                f"Restart {restart} pass {pass_index} color {color}: "
                f"candidates={candidate_count}, edges={left_edges.size}, "
                f"selected={selected_count}, time={perf_counter() - edge_start:.2f}s",
                flush=True,
            )

        current_bonus = int((active_mask & (grid == bonus)).sum().item())
        print(
            f"Restart {restart} pass {pass_index}: fraction={fraction:.4g}, "
            f"candidates={pass_candidates}, selected={pass_assigned}, "
            f"edges={pass_edges}, active_bonus={current_bonus}",
            flush=True,
        )
        if current_bonus == previous_bonus and pass_assigned == 0:
            break
        previous_bonus = current_bonus

    return grid


def fill_active_cells(base_grid, active_mask, colors, mask_kernel, pad, mask_offsets, restart):
    if args.fill_strategy == "isolated":
        return fill_active_cells_isolated(
            base_grid,
            active_mask,
            colors,
            mask_kernel,
            pad,
            restart,
        )
    return fill_active_cells_mis(
        base_grid,
        active_mask,
        colors,
        mask_kernel,
        pad,
        mask_offsets,
        restart,
    )


def gather_single_blockers(grid, candidate_mask, target_color, mask_offsets):
    coords = torch.nonzero(candidate_mask, as_tuple=False)
    num_candidates = int(coords.shape[0])
    if num_candidates == 0:
        return coords, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    height, width = grid.shape
    offsets = torch.as_tensor(
        unique_offsets(mask_offsets),
        dtype=torch.long,
        device=grid.device,
    )
    edge_chunk_size = max(1, int(args.mis_edge_chunk_size))
    ci = coords[:, 0]
    cj = coords[:, 1]
    blocker_i = torch.full(
        (num_candidates,),
        -1,
        dtype=torch.long,
        device=grid.device,
    )
    blocker_j = torch.full(
        (num_candidates,),
        -1,
        dtype=torch.long,
        device=grid.device,
    )

    for chunk_start in range(0, int(offsets.shape[0]), edge_chunk_size):
        offset_chunk = offsets[chunk_start:chunk_start + edge_chunk_size]
        ni = (ci.unsqueeze(1) + offset_chunk[:, 0].unsqueeze(0)) % height
        nj = (cj.unsqueeze(1) + offset_chunk[:, 1].unsqueeze(0)) % width
        matches = grid[ni, nj] == target_color
        if not matches.any():
            continue

        hit_candidate, hit_offset = torch.nonzero(matches, as_tuple=True)
        blocker_i[hit_candidate] = ni[hit_candidate, hit_offset]
        blocker_j[hit_candidate] = nj[hit_candidate, hit_offset]

    found = blocker_i >= 0
    if not found.all():
        coords = coords[found]
        blocker_i = blocker_i[found]
        blocker_j = blocker_j[found]

    return (
        coords,
        blocker_i.detach().cpu().numpy(),
        blocker_j.detach().cpu().numpy(),
    )


def first_available_recolor(blocker_counts, current_color, rng):
    available = blocker_counts == 0
    available[:, current_color] = False
    has_alternative = available.any(axis=1)
    choices = np.full(available.shape[0], -1, dtype=np.int64)
    if has_alternative.any():
        jitter = rng.random(available.shape, dtype=np.float32)
        jitter[~available] = -1.0
        choices[has_alternative] = jitter[has_alternative].argmax(axis=1)
    return choices


def choose_unique_blocker_proposals(blocker_i, blocker_j, width, rng):
    num_proposals = int(blocker_i.shape[0])
    if num_proposals == 0:
        return np.empty(0, dtype=np.int64)

    order = rng.permutation(num_proposals)
    blocker_flat = blocker_i.astype(np.int64, copy=False) * int(width) + blocker_j
    _, first_positions = np.unique(blocker_flat[order], return_index=True)
    chosen = order[first_positions]
    max_proposals = int(args.augment_max_proposals_per_color)
    if max_proposals > 0 and chosen.shape[0] > max_proposals:
        chosen = chosen[:max_proposals]
    return chosen


def propose_single_blocker_moves(current, active_mask, counts, target_color, mask_offsets):
    bonus = counts.shape[0] - 1
    candidate_mask = active_mask & (current == bonus) & (counts[target_color] == 1)
    if not candidate_mask.any():
        return None

    coords, blocker_i, blocker_j = gather_single_blockers(
        current,
        candidate_mask,
        target_color,
        mask_offsets,
    )
    if int(coords.shape[0]) == 0:
        return None

    coords_cpu = coords.detach().cpu().numpy()
    blocker_active = active_mask[blocker_i, blocker_j].detach().cpu().numpy()
    blocker_has_target = (
        current[blocker_i, blocker_j].detach().cpu().numpy() == target_color
    )
    keep = blocker_active & blocker_has_target
    if not keep.any():
        return None

    return coords_cpu[keep], blocker_i[keep], blocker_j[keep]


def augment_single_blocker_swaps(grid, active_mask, colors, mask_kernel, pad, mask_offsets, restart):
    if args.augment_rounds <= 0:
        return grid

    bonus = colors - 1
    height, width = grid.shape
    rng = np.random.default_rng(args.seed + restart * 1000003 + 424242)
    current = grid.clone()
    previous_bonus = int((active_mask & (current == bonus)).sum().item())

    for round_index in range(args.augment_rounds):
        round_accepted = 0
        round_proposed = 0
        for target_color in color_order_for_pass(bonus, restart, 1000 + round_index):
            counts = count_same_color_neighbors(
                one_hot_encode(current, colors),
                mask_kernel,
                pad,
            )
            proposed_moves = propose_single_blocker_moves(
                current,
                active_mask,
                counts,
                target_color,
                mask_offsets,
            )
            if proposed_moves is None:
                continue
            coords_cpu, blocker_i, blocker_j = proposed_moves
            blocker_i_t = torch.as_tensor(blocker_i, dtype=torch.long, device=current.device)
            blocker_j_t = torch.as_tensor(blocker_j, dtype=torch.long, device=current.device)
            blocker_counts = (
                counts[:bonus, blocker_i_t, blocker_j_t]
                .transpose(0, 1)
                .detach()
                .cpu()
                .numpy()
            )
            recolors = first_available_recolor(blocker_counts, target_color, rng)
            keep = recolors >= 0
            if not keep.any():
                continue

            coords_cpu = coords_cpu[keep]
            blocker_i = blocker_i[keep]
            blocker_j = blocker_j[keep]
            recolors = recolors[keep]
            chosen = choose_unique_blocker_proposals(blocker_i, blocker_j, width, rng)
            if chosen.size == 0:
                continue

            coords_cpu = coords_cpu[chosen]
            blocker_i = blocker_i[chosen]
            blocker_j = blocker_j[chosen]
            recolors = recolors[chosen]

            u_i = torch.as_tensor(coords_cpu[:, 0], dtype=torch.long, device=current.device)
            u_j = torch.as_tensor(coords_cpu[:, 1], dtype=torch.long, device=current.device)
            b_i = torch.as_tensor(blocker_i, dtype=torch.long, device=current.device)
            b_j = torch.as_tensor(blocker_j, dtype=torch.long, device=current.device)
            recolor_t = torch.as_tensor(recolors, dtype=torch.long, device=current.device)

            current[u_i, u_j] = target_color
            current[b_i, b_j] = recolor_t
            conflicts = real_conflict_mask(current, colors, mask_kernel, pad)
            bad = conflicts[u_i, u_j] | conflicts[b_i, b_j]
            bad_count = int(bad.sum().item())
            if bad_count:
                current[u_i[bad], u_j[bad]] = bonus
                current[b_i[bad], b_j[bad]] = target_color

            accepted = int(chosen.size) - bad_count
            round_accepted += accepted
            round_proposed += int(chosen.size)
            print(
                f"Restart {restart} augment {round_index} color {target_color}: "
                f"proposed={chosen.size}, accepted={accepted}, reverted={bad_count}",
                flush=True,
            )

        current_bonus = int((active_mask & (current == bonus)).sum().item())
        print(
            f"Restart {restart} augment {round_index}: proposed={round_proposed}, "
            f"accepted={round_accepted}, active_bonus={current_bonus}",
            flush=True,
        )
        if round_accepted == 0 or current_bonus >= previous_bonus:
            break
        previous_bonus = current_bonus

    return current


def kick_single_blockers(grid, active_mask, colors, mask_kernel, pad, mask_offsets, restart, round_index):
    bonus = colors - 1
    height, width = grid.shape
    rng = np.random.default_rng(args.seed + restart * 1000003 + round_index * 9176 + 777777)
    current = grid.clone()
    total_proposed = 0
    total_accepted = 0

    for target_color in color_order_for_pass(bonus, restart, 2000 + round_index):
        counts = count_same_color_neighbors(
            one_hot_encode(current, colors),
            mask_kernel,
            pad,
        )
        proposed_moves = propose_single_blocker_moves(
            current,
            active_mask,
            counts,
            target_color,
            mask_offsets,
        )
        if proposed_moves is None:
            continue

        coords_cpu, blocker_i, blocker_j = proposed_moves
        chosen = choose_unique_blocker_proposals(blocker_i, blocker_j, width, rng)
        max_proposals = int(args.kick_max_proposals_per_color)
        if max_proposals > 0 and chosen.shape[0] > max_proposals:
            chosen = chosen[:max_proposals]
        if chosen.size == 0:
            continue

        coords_cpu = coords_cpu[chosen]
        blocker_i = blocker_i[chosen]
        blocker_j = blocker_j[chosen]

        u_i = torch.as_tensor(coords_cpu[:, 0], dtype=torch.long, device=current.device)
        u_j = torch.as_tensor(coords_cpu[:, 1], dtype=torch.long, device=current.device)
        b_i = torch.as_tensor(blocker_i, dtype=torch.long, device=current.device)
        b_j = torch.as_tensor(blocker_j, dtype=torch.long, device=current.device)

        current[u_i, u_j] = target_color
        current[b_i, b_j] = bonus
        conflicts = real_conflict_mask(current, colors, mask_kernel, pad)
        bad = conflicts[u_i, u_j]
        bad_count = int(bad.sum().item())
        if bad_count:
            current[u_i[bad], u_j[bad]] = bonus
            current[b_i[bad], b_j[bad]] = target_color

        accepted = int(chosen.size) - bad_count
        total_proposed += int(chosen.size)
        total_accepted += accepted
        print(
            f"Restart {restart} kick {round_index} color {target_color}: "
            f"proposed={chosen.size}, accepted={accepted}, reverted={bad_count}",
            flush=True,
        )

    return current, total_proposed, total_accepted


def kick_and_refill(grid, active_mask, colors, mask_kernel, pad, mask_offsets, restart):
    if args.kick_rounds <= 0:
        return grid

    bonus = colors - 1
    current = grid.clone()
    for round_index in range(args.kick_rounds):
        before = current.clone()
        before_bonus = int((active_mask & (before == bonus)).sum().item())
        kicked, proposed, accepted = kick_single_blockers(
            current,
            active_mask,
            colors,
            mask_kernel,
            pad,
            mask_offsets,
            restart,
            round_index,
        )
        if accepted == 0:
            print(
                f"Restart {restart} kick {round_index}: no accepted kicks",
                flush=True,
            )
            break

        refilled = fill_active_cells(
            kicked,
            active_mask,
            colors,
            mask_kernel,
            pad,
            mask_offsets,
            restart + 1000 + round_index,
        )
        after_bonus = int((active_mask & (refilled == bonus)).sum().item())
        print(
            f"Restart {restart} kick {round_index}: proposed={proposed}, "
            f"accepted={accepted}, before_bonus={before_bonus}, "
            f"after_refill_bonus={after_bonus}",
            flush=True,
        )
        if after_bonus < before_bonus:
            current = refilled
        else:
            current = before
            break

    return current


def save_summary(summary, final_grid):
    fixed_dir = output_dir / "fixed_colorings"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    npy_path = fixed_dir / f"{args.run_id}_{args.eval_gridsize}_heuristic_fixed.npy"
    np.save(npy_path, final_grid.astype(np.uint8, copy=False))

    json_path = output_dir / "heuristic_summary.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary | {"npy_path": str(npy_path)}, handle, indent=2, sort_keys=True)

    csv_path = output_dir / "verified_paralellograms_heuristic.csv"
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "eval_gridsize",
                "fraction_fixed_to_bonus (%)",
                "real_conflict_cells",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": args.run_id,
                "eval_gridsize": args.eval_gridsize,
                "fraction_fixed_to_bonus (%)": round(summary["bonus_fraction_percent"], 8),
                "real_conflict_cells": summary["real_conflict_cells"],
            }
        )

    print(f"Saved heuristic coloring to {npy_path}", flush=True)
    print(f"Saved heuristic summary to {json_path}", flush=True)


def main():
    with open(args.config_json, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, parallelogram = load_model(config, args.checkpoint_path)
    colors = int(config["n_colours"])
    gridsize = int(args.eval_gridsize)

    print(f"device={device}", flush=True)
    print(f"run_id={args.run_id}", flush=True)
    print(f"eval_gridsize={gridsize}", flush=True)
    print(f"passes={args.passes}", flush=True)
    print(f"restarts={args.restarts}", flush=True)
    print(f"fill_strategy={args.fill_strategy}", flush=True)
    print(f"augment_rounds={args.augment_rounds}", flush=True)
    print(f"kick_rounds={args.kick_rounds}", flush=True)
    print(f"candidate_fractions={candidate_fractions}", flush=True)

    mask_start = perf_counter()
    starts, base_mask = compute_base_mask_vec(parallelogram, gridsize)
    print(f"Prepared base mask and grid starts in {perf_counter() - mask_start:.2f}s", flush=True)

    eval_start = perf_counter()
    with torch.no_grad():
        outputs = model(starts.to(device)).argmax(dim=1)
        initial_grid = outputs.reshape(gridsize, gridsize).long()
    print(f"Evaluated model on verifier grid in {perf_counter() - eval_start:.2f}s", flush=True)

    mask_offsets = np.asarray(base_mask, dtype=np.int64)
    print(
        f"Computed base mask. len(base_mask)={len(base_mask)}, "
        f"first={base_mask[0] if base_mask else None}, shape={mask_offsets.shape}",
        flush=True,
    )

    kernel_start = perf_counter()
    if args.neighbor_backend == "fft":
        mask_kernel = build_fft_mask_kernel(mask_offsets, gridsize, gridsize)
        pad = None
    else:
        mask_kernel, pad = build_mask_kernel(mask_offsets)
    print(f"Prepared neighbor backend in {perf_counter() - kernel_start:.2f}s", flush=True)

    grid, conflicts = apply_initial_greedy_repairs(initial_grid.clone(), colors, mask_kernel, pad)
    active_mask = (grid == colors - 1) | conflicts
    active_count = int(active_mask.sum().item())

    base_grid = grid.clone()
    base_grid[active_mask] = colors - 1
    base_conflicts = real_conflict_mask(base_grid, colors, mask_kernel, pad)
    base_conflict_cells = int(base_conflicts.sum().item())
    print(
        f"Active cells after pre-repair: {active_count} out of {gridsize * gridsize}",
        flush=True,
    )
    print(
        f"All-active-bonus baseline: bonus_fraction={bonus_fraction(base_grid, colors) * 100:.8f}%, "
        f"real_conflict_cells={base_conflict_cells}",
        flush=True,
    )
    if base_conflict_cells:
        raise RuntimeError(
            f"Internal baseline is invalid: {base_conflict_cells} frozen real conflict cells remain."
        )

    best_grid_cpu = None
    best_fraction = float("inf")
    for restart in range(args.restarts):
        restart_grid = fill_active_cells(
            base_grid,
            active_mask,
            colors,
            mask_kernel,
            pad,
            mask_offsets,
            restart,
        )
        restart_grid = augment_single_blocker_swaps(
            restart_grid,
            active_mask,
            colors,
            mask_kernel,
            pad,
            mask_offsets,
            restart,
        )
        restart_grid = kick_and_refill(
            restart_grid,
            active_mask,
            colors,
            mask_kernel,
            pad,
            mask_offsets,
            restart,
        )
        restart_conflicts = real_conflict_mask(restart_grid, colors, mask_kernel, pad)
        conflict_cells = int(restart_conflicts.sum().item())
        restart_fraction = bonus_fraction(restart_grid, colors)
        print(
            f"Restart {restart} final: bonus_fraction={restart_fraction * 100:.8f}%, "
            f"real_conflict_cells={conflict_cells}",
            flush=True,
        )
        if conflict_cells == 0 and restart_fraction < best_fraction:
            best_fraction = restart_fraction
            best_grid_cpu = restart_grid.detach().cpu().to(torch.uint8).numpy()

    if best_grid_cpu is None:
        raise RuntimeError("No restart produced a conflict-free coloring.")

    final_grid = torch.from_numpy(best_grid_cpu.astype(np.int64)).to(device)
    final_conflicts = real_conflict_mask(final_grid, colors, mask_kernel, pad)
    final_conflict_cells = int(final_conflicts.sum().item())
    final_fraction = float(np.count_nonzero(best_grid_cpu == colors - 1)) / float(best_grid_cpu.size)
    print(f"Final real-color conflict cells: {final_conflict_cells}", flush=True)
    print(f"Fraction set to bonus color: {final_fraction * 100:.8f}%", flush=True)
    print(f"Fraction set to {colors - 1}: {final_fraction * 100:.8f}%", flush=True)
    if final_conflict_cells:
        raise RuntimeError(f"Heuristic coloring has {final_conflict_cells} real-color conflicts.")

    summary = {
        "active_cells": active_count,
        "baseline_bonus_fraction_percent": bonus_fraction(base_grid, colors) * 100.0,
        "bonus_fraction_percent": final_fraction * 100.0,
        "candidate_fractions": candidate_fractions,
        "eval_gridsize": gridsize,
        "fill_strategy": args.fill_strategy,
        "augment_rounds": args.augment_rounds,
        "kick_rounds": args.kick_rounds,
        "passes": args.passes,
        "real_conflict_cells": final_conflict_cells,
        "restarts": args.restarts,
        "run_id": args.run_id,
        "seed": args.seed,
    }
    save_summary(summary, best_grid_cpu)

    if not args.no_plot:
        print("Plotting is intentionally omitted in the heuristic script; use --no_plot.", flush=True)


if __name__ == "__main__":
    main()
