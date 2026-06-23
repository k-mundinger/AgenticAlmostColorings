#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from ensemble_eval import evaluate_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble sweep runs on parallelogram bonus/conflict metrics."
    )
    parser.add_argument("sweep_root", type=Path, help="Directory containing per-run output directories.")
    parser.add_argument("--output-csv", type=Path, default=None, help="CSV path (default: sweep_root/summary_eval.csv).")
    parser.add_argument("--top-k", type=int, default=20, help="Rows to print.")
    parser.add_argument("--eval-grid-size", type=int, default=128)
    parser.add_argument("--eval-circle-points", type=int, default=128)
    parser.add_argument("--device", default=None, help="cuda:0 or cpu (auto-detect by default).")
    return parser.parse_args()


def read_losses(path: Path) -> tuple[float | None, float | None]:
    best_loss = None
    final_loss = None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loss = float(row["loss"])
            final_loss = loss
            if best_loss is None or loss < best_loss:
                best_loss = loss
    return best_loss, final_loss


def main() -> int:
    args = parse_args()
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rows = []
    for run_dir in sorted(args.sweep_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "logs":
            continue
        if not (run_dir / "trained_model.pt").exists():
            continue

        row = {"run_id": run_dir.name, "run_dir": str(run_dir)}
        loss_path = run_dir / "train_losses.csv"
        if loss_path.exists():
            best_loss, final_loss = read_losses(loss_path)
            row["best_loss"] = best_loss
            row["final_loss"] = final_loss

        metrics = evaluate_run_dir(
            run_dir=run_dir,
            eval_grid_size=args.eval_grid_size,
            eval_circle_points=args.eval_circle_points,
            device=device,
        )
        row.update(metrics)
        rows.append(row)

    if not rows:
        raise SystemExit(f"No evaluable runs found under {args.sweep_root}")

    rows.sort(
        key=lambda row: (
            row["bad_fraction_pct"],
            row["bonus_fraction_pct"],
            row["real_conflict_fraction_pct"],
            row["run_id"],
        )
    )

    output_csv = args.output_csv or (args.sweep_root / "summary_eval.csv")
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows[: args.top_k])
    print(f"Wrote {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
