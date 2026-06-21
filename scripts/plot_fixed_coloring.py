#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "aac_mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a saved verifier coloring without rerunning verification."
    )
    parser.add_argument("--coloring-npy", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--parallelogram-max-cells", type=int, default=1024)
    parser.add_argument(
        "--formats",
        default="png",
        help="Comma-separated output formats, e.g. png,pdf,svg.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def get_cmap():
    return ListedColormap(
        [
            "#FFD6A5",
            "#FDFFB6",
            "#CAFFBF",
            "#9BF6FF",
            "#A0C4FF",
            "#FFADAD",
        ],
        name="aac_pastel",
    )


def downsample_nearest(grid, max_cells):
    if max_cells <= 0 or max(grid.shape) <= max_cells:
        return grid, 1
    step = int(np.ceil(max(grid.shape) / max_cells))
    return grid[::step, ::step], step


def parse_formats(raw):
    formats = []
    for item in raw.split(","):
        fmt = item.strip().lower().lstrip(".")
        if fmt:
            formats.append(fmt)
    if not formats:
        raise ValueError("At least one output format is required.")
    return formats


def save_array_image(array, output_path, cmap, vmin, vmax, dpi):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", rasterized=True)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_parallelogram_coloring(grid, parallelogram, output_path, n_colours, dpi):
    rows, cols = grid.shape
    v1 = np.asarray(parallelogram[0], dtype=np.float64)
    v2 = np.asarray(parallelogram[1], dtype=np.float64)

    a = np.linspace(0.0, 1.0, rows + 1)
    b = np.linspace(0.0, 1.0, cols + 1)
    aa, bb = np.meshgrid(a, b, indexing="ij")
    x = aa * v1[0] + bb * v2[0]
    y = aa * v1[1] + bb * v2[1]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax.pcolormesh(
        x,
        y,
        grid,
        cmap=get_cmap(),
        vmin=0,
        vmax=n_colours - 1,
        shading="flat",
        linewidth=0,
        rasterized=True,
    )

    corners = np.array(
        [
            [0.0, 0.0],
            v1,
            v1 + v2,
            v2,
            [0.0, 0.0],
        ]
    )
    ax.plot(corners[:, 0], corners[:, 1], color="black", linewidth=0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    args = parse_args()

    coloring_path = Path(args.coloring_npy)
    config_path = Path(args.config_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    grid = np.load(coloring_path)
    n_colours = int(config["n_colours"])
    bonus_colour = n_colours - 1
    prefix = args.prefix or coloring_path.stem

    formats = parse_formats(args.formats)
    plot_grid, step = downsample_nearest(grid, args.parallelogram_max_cells)
    saved_paths = []
    for fmt in formats:
        square_path = output_dir / f"{prefix}_square.{fmt}"
        bonus_path = output_dir / f"{prefix}_bonus_mask.{fmt}"
        parallelogram_path = output_dir / f"{prefix}_parallelogram.{fmt}"

        save_array_image(grid, square_path, get_cmap(), 0, n_colours - 1, args.dpi)
        save_array_image((grid == bonus_colour).astype(np.uint8), bonus_path, "gray", 0, 1, args.dpi)
        save_parallelogram_coloring(
            plot_grid,
            config["training"]["parallelogram"],
            parallelogram_path,
            n_colours,
            args.dpi,
        )
        saved_paths.extend(
            [
                ("square", square_path),
                ("bonus_mask", bonus_path),
                ("parallelogram", parallelogram_path),
            ]
        )

    bonus_fraction = float(np.count_nonzero(grid == bonus_colour)) / float(grid.size) * 100.0
    print(f"grid_shape={grid.shape}")
    print(f"bonus_fraction_percent={bonus_fraction:.8f}")
    print(f"parallelogram_downsample_step={step}")
    for label, path in saved_paths:
        print(f"{label}={path}")


if __name__ == "__main__":
    main()
