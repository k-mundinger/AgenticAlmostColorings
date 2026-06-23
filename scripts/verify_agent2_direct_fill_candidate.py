#!/usr/bin/env python3
"""Independently verify agent 2's paper-grid direct-fill candidate.

The candidate is a compact CSV diff that recolors selected bonus cells (color 5)
to real colors 0..4. This verifier intentionally does not import the agent's
generated analysis script. It rebuilds the rectangular torus unit-distance mask
from the paper parallelogram, validates the diff against the original grid, and
recomputes same-color conflicts for the original and patched grids.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

BONUS_COLOR = 5
REAL_COLORS = tuple(range(5))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_parallelogram(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(newline="")))
    by_id = {row["vector_id"]: row for row in rows}
    required = {"v1", "v2"}
    missing = required - set(by_id)
    if missing:
        raise ValueError(f"missing parallelogram rows: {sorted(missing)}")
    return {
        "v1": np.array([float(by_id["v1"]["x"]), float(by_id["v1"]["y"])], dtype=np.float64),
        "v2": np.array([float(by_id["v2"]["x"]), float(by_id["v2"]["y"])], dtype=np.float64),
        "n1": int(by_id["v1"]["subdivisions"]),
        "n2": int(by_id["v2"]["subdivisions"]),
        "raw_rows": rows,
    }


def load_colors(grid_csv: Path, h: int, w: int) -> np.ndarray:
    flat = np.loadtxt(grid_csv, delimiter=",", skiprows=1, usecols=2, dtype=np.uint8)
    expected = h * w
    if flat.size != expected:
        raise ValueError(f"grid row count {flat.size} != expected {expected}")
    colors = flat.reshape(h, w)
    bad = colors[(colors < 0) | (colors > BONUS_COLOR)]
    if bad.size:
        raise ValueError(f"grid contains colors outside 0..{BONUS_COLOR}")
    return colors


def read_diff(diff_csv: Path, v1: np.ndarray, v2: np.ndarray, h: int, w: int) -> list[dict]:
    changes: list[dict] = []
    seen: set[tuple[int, int]] = set()
    dv1 = v1 / h
    dv2 = v2 / w
    with diff_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"i", "j", "x", "y", "old", "new"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"diff CSV missing columns: {sorted(missing)}")
        for row_num, row in enumerate(reader, start=2):
            i = int(row["i"])
            j = int(row["j"])
            old = int(row["old"])
            new = int(row["new"])
            if not (0 <= i < h and 0 <= j < w):
                raise ValueError(f"diff row {row_num}: index out of bounds {(i, j)} for {h}x{w}")
            if (i, j) in seen:
                raise ValueError(f"diff row {row_num}: duplicate cell {(i, j)}")
            if old != BONUS_COLOR or new not in REAL_COLORS:
                raise ValueError(f"diff row {row_num}: expected old=5 and new in 0..4, got old={old}, new={new}")
            x = float(row["x"])
            y = float(row["y"])
            expected = i * dv1 + j * dv2
            coord_err = max(abs(x - float(expected[0])), abs(y - float(expected[1])))
            changes.append(
                {
                    "row_num": row_num,
                    "i": i,
                    "j": j,
                    "old": old,
                    "new": new,
                    "x": x,
                    "y": y,
                    "coord_err": float(coord_err),
                }
            )
            seen.add((i, j))
    if not changes:
        raise ValueError("diff CSV is empty")
    return changes


def build_unit_distance_mask(v1: np.ndarray, v2: np.ndarray, h: int, w: int, radius: float) -> np.ndarray:
    """Build the conservative cell-offset mask using a NumPy implementation.

    A target cell offset is active when one of that cell's translated copies is
    cut by a radius-1 circle around any corner of the origin cell. Offsets are
    recorded modulo the rectangular torus, matching the repository verifier's
    FFT-kernel convention.
    """

    dv1 = v1 / h
    dv2 = v2 / w
    ii = np.arange(h, dtype=np.float64)[:, None]
    jj = np.arange(w, dtype=np.float64)[None, :]
    starts_x = ii * dv1[0] + jj * dv2[0]
    starts_y = ii * dv1[1] + jj * dv2[1]

    corner_offsets = np.array(
        [
            [0.0, 0.0],
            dv1,
            dv1 + dv2,
            dv2,
        ],
        dtype=np.float64,
    )
    radius_sq = float(radius * radius)
    active_parts: list[np.ndarray] = []
    for tile_i in (-1, 0, 1):
        for tile_j in (-1, 0, 1):
            shift = tile_i * v1 + tile_j * v2
            tile_partial = np.zeros((h, w), dtype=bool)
            for center in corner_offsets:
                inside_any = np.zeros((h, w), dtype=bool)
                inside_all = np.ones((h, w), dtype=bool)
                for corner in corner_offsets:
                    dx = starts_x + shift[0] + corner[0] - center[0]
                    dy = starts_y + shift[1] + corner[1] - center[1]
                    inside = (dx * dx + dy * dy) <= radius_sq
                    inside_any |= inside
                    inside_all &= inside
                tile_partial |= inside_any & (~inside_all)
            coords = np.argwhere(tile_partial)
            if coords.size:
                active_parts.append(coords.astype(np.int64, copy=False))
    if not active_parts:
        return np.empty((0, 2), dtype=np.int64)
    mask = np.concatenate(active_parts, axis=0)
    mask[:, 0] %= h
    mask[:, 1] %= w
    return np.unique(mask, axis=0)


def directed_conflicts_by_color(colors: np.ndarray, mask: np.ndarray) -> dict:
    h, w = colors.shape
    kernel = np.zeros((h, w), dtype=np.float64)
    for di, dj in mask:
        kernel[(-int(di)) % h, (-int(dj)) % w] = 1.0
    kernel_fft = np.fft.rfft2(kernel)
    by_color: dict[str, int] = {}
    for color in REAL_COLORS:
        indicator = (colors == color).astype(np.float64)
        neighbor_counts = np.fft.irfft2(np.fft.rfft2(indicator) * kernel_fft, s=(h, w))
        counts = np.rint(neighbor_counts)
        directed = int(counts[colors == color].sum())
        by_color[str(color)] = directed
    return by_color


def validate_and_apply_diff(colors: np.ndarray, changes: list[dict]) -> np.ndarray:
    patched = colors.copy()
    for change in changes:
        i = change["i"]
        j = change["j"]
        old = change["old"]
        new = change["new"]
        actual = int(patched[i, j])
        if actual != old:
            raise ValueError(f"diff row {change['row_num']}: grid[{i},{j}]={actual}, expected old={old}")
        patched[i, j] = new
    return patched


def local_changed_cell_conflicts(patched: np.ndarray, changes: list[dict], mask: np.ndarray, limit: int = 50) -> dict:
    h, w = patched.shape
    offsets: set[tuple[int, int]] = set()
    for di, dj in mask:
        a = int(di) % h
        b = int(dj) % w
        offsets.add((a, b))
        offsets.add((-a % h, -b % w))

    examples = []
    total = 0
    for change in changes:
        i = change["i"]
        j = change["j"]
        color = change["new"]
        for di, dj in offsets:
            ni = (i + di) % h
            nj = (j + dj) % w
            if ni == i and nj == j:
                continue
            if int(patched[ni, nj]) == color:
                total += 1
                if len(examples) < limit:
                    examples.append(
                        {
                            "changed_cell": [i, j],
                            "neighbor": [int(ni), int(nj)],
                            "offset": [int(di), int(dj)],
                            "color": int(color),
                        }
                    )
    return {"conflict_checks": len(changes) * len(offsets), "conflict_count": total, "examples": examples}


def color_counts(colors: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(colors, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-csv", type=Path, required=True)
    parser.add_argument("--parallelogram-csv", type=Path, required=True)
    parser.add_argument("--diff-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--skip-full-fft", action="store_true")
    args = parser.parse_args()

    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    par = read_parallelogram(args.parallelogram_csv)
    h = par["n1"]
    w = par["n2"]
    v1 = par["v1"]
    v2 = par["v2"]

    print(f"loading grid {args.grid_csv} as {h}x{w}", flush=True)
    colors = load_colors(args.grid_csv, h, w)
    changes = read_diff(args.diff_csv, v1, v2, h, w)
    patched = validate_and_apply_diff(colors, changes)

    print(f"building unit-distance mask for {h}x{w}", flush=True)
    mask_start = time.time()
    mask = build_unit_distance_mask(v1, v2, h, w, args.radius)
    mask_seconds = time.time() - mask_start
    np.save(args.output_dir / "independent_unit_distance_mask.npy", mask.astype(np.int32))
    print(f"mask_size={len(mask)} mask_seconds={mask_seconds:.2f}", flush=True)

    local = local_changed_cell_conflicts(patched, changes, mask)
    print(f"local_changed_cell_conflicts={local['conflict_count']}", flush=True)

    baseline_conflicts = None
    patched_conflicts = None
    fft_seconds = None
    if not args.skip_full_fft:
        fft_start = time.time()
        print("running full-grid FFT conflict check for original grid", flush=True)
        baseline_conflicts = directed_conflicts_by_color(colors, mask)
        print("running full-grid FFT conflict check for patched grid", flush=True)
        patched_conflicts = directed_conflicts_by_color(patched, mask)
        fft_seconds = time.time() - fft_start
        print(f"fft_seconds={fft_seconds:.2f}", flush=True)

    old_counts = color_counts(colors)
    new_counts = color_counts(patched)
    max_coord_err = max(change["coord_err"] for change in changes)
    result = {
        "success": bool(
            local["conflict_count"] == 0
            and (baseline_conflicts is None or sum(baseline_conflicts.values()) == 0)
            and (patched_conflicts is None or sum(patched_conflicts.values()) == 0)
            and int(old_counts.get(str(BONUS_COLOR), 0)) - int(new_counts.get(str(BONUS_COLOR), 0)) == len(changes)
        ),
        "grid_csv": str(args.grid_csv),
        "grid_sha256": sha256_file(args.grid_csv),
        "parallelogram_csv": str(args.parallelogram_csv),
        "parallelogram_sha256": sha256_file(args.parallelogram_csv),
        "diff_csv": str(args.diff_csv),
        "diff_sha256": sha256_file(args.diff_csv),
        "shape": [h, w],
        "radius": args.radius,
        "mask_size": int(len(mask)),
        "mask_seconds": mask_seconds,
        "changes": len(changes),
        "max_diff_coordinate_error": float(max_coord_err),
        "old_color_counts": old_counts,
        "patched_color_counts": new_counts,
        "bonus_count_before": int(old_counts.get(str(BONUS_COLOR), 0)),
        "bonus_count_after": int(new_counts.get(str(BONUS_COLOR), 0)),
        "bonus_fraction_before": float(old_counts.get(str(BONUS_COLOR), 0) / colors.size),
        "bonus_fraction_after": float(new_counts.get(str(BONUS_COLOR), 0) / colors.size),
        "bonus_reduction": int(old_counts.get(str(BONUS_COLOR), 0)) - int(new_counts.get(str(BONUS_COLOR), 0)),
        "local_changed_cell_check": local,
        "baseline_directed_conflicts_by_color": baseline_conflicts,
        "patched_directed_conflicts_by_color": patched_conflicts,
        "baseline_directed_conflicts_total": None if baseline_conflicts is None else int(sum(baseline_conflicts.values())),
        "patched_directed_conflicts_total": None if patched_conflicts is None else int(sum(patched_conflicts.values())),
        "fft_seconds": fft_seconds,
        "elapsed_seconds": time.time() - started,
    }

    out_json = args.output_dir / "agent2_direct_fill_independent_verification.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True))
    summary = args.output_dir / "summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"success={result['success']}",
                f"mask_size={result['mask_size']}",
                f"changes={result['changes']}",
                f"bonus_fraction_before={result['bonus_fraction_before']:.12f}",
                f"bonus_fraction_after={result['bonus_fraction_after']:.12f}",
                f"local_changed_cell_conflicts={local['conflict_count']}",
                f"baseline_directed_conflicts_total={result['baseline_directed_conflicts_total']}",
                f"patched_directed_conflicts_total={result['patched_directed_conflicts_total']}",
                f"json={out_json}",
            ]
        )
        + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
