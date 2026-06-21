#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fast training sweep losses.")
    parser.add_argument("sweep_root", type=Path, help="Directory containing per-run output directories.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path to write a CSV summary.")
    parser.add_argument("--top-k", type=int, default=20, help="Rows to print, sorted by best_loss.")
    return parser.parse_args()


def read_losses(path: Path) -> tuple[int | None, float | None, int | None, float | None]:
    best_step = None
    best_loss = None
    final_step = None
    final_loss = None

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            loss = float(row["loss"])
            final_step = step
            final_loss = loss
            if best_loss is None or loss < best_loss:
                best_step = step
                best_loss = loss

    return best_step, best_loss, final_step, final_loss


def main() -> int:
    args = parse_args()
    rows = []

    for run_dir in sorted(args.sweep_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "logs":
            continue
        loss_path = run_dir / "train_losses.csv"
        if not loss_path.exists():
            continue
        best_step, best_loss, final_step, final_loss = read_losses(loss_path)
        rows.append(
            {
                "run_id": run_dir.name,
                "best_step": best_step,
                "best_loss": best_loss,
                "final_step": final_step,
                "final_loss": final_loss,
                "has_model": (run_dir / "trained_model.pt").exists(),
                "run_dir": str(run_dir),
            }
        )

    rows.sort(key=lambda row: (float("inf") if row["best_loss"] is None else row["best_loss"], row["run_id"]))

    fieldnames = ["run_id", "best_step", "best_loss", "final_step", "final_loss", "has_model", "run_dir"]
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows[: args.top_k])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
