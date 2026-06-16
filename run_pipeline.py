#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RUN_DIR_RE = re.compile(r"Saved persistent local copy of all outputs to models/([A-Za-z0-9_-]+)")
BONUS_FRAC_RE = re.compile(r"Fraction set to bonus color:\s*([0-9.]+)%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a model and automatically run ILP verification."
    )
    parser.add_argument(
        "--train-command",
        default="python main.py --debug",
        help="Training command to execute.",
    )
    parser.add_argument(
        "--eval-gridsize",
        type=int,
        default=32,
        help="Grid size for ILP verification (default: coarse 32x32).",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable W&B integration for training/verifier API lookups.",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT", "2DHadwigerNelson"),
        help="W&B project used by verifier when --use-wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.getenv("WANDB_ENTITY", "ais2t"),
        help="W&B entity used by verifier when --use-wandb is enabled.",
    )
    parser.add_argument(
        "--config-overrides-json",
        default=None,
        help="Optional JSON file merged into local default config before verification.",
    )
    return parser.parse_args()


def run_command(command: List[str], env: Dict[str, str]) -> Tuple[int, str]:
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    output_lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)
    return proc.wait(), "".join(output_lines)


def parse_run_id(training_output: str, preexisting_runs: set[str], models_dir: Path) -> Optional[str]:
    match = RUN_DIR_RE.search(training_output)
    if match:
        return match.group(1)

    current_runs = {p.name for p in models_dir.iterdir() if p.is_dir()} if models_dir.exists() else set()
    new_runs = sorted(current_runs - preexisting_runs)
    if new_runs:
        newest = max((models_dir / run_id for run_id in new_runs), key=lambda p: p.stat().st_mtime)
        return newest.name

    if current_runs:
        newest = max((models_dir / run_id for run_id in current_runs), key=lambda p: p.stat().st_mtime)
        return newest.name
    return None


def load_defaults_from_main(main_path: Path) -> Dict[str, Any]:
    tree = ast.parse(main_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "defaults":
                    # `defaults` is defined as dict(...), which literal_eval does
                    # not support. Evaluate with a tiny, safe namespace.
                    expr = ast.Expression(body=node.value)
                    code = compile(expr, filename=str(main_path), mode="eval")
                    return eval(code, {"__builtins__": {}, "dict": dict}, {})
    raise RuntimeError("Could not parse `defaults` dictionary from main.py")


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_checkpoint(run_dir: Path) -> Optional[Path]:
    for candidate in ("trained_model.pt", "step_32768_model.pt", "step_65536_model.pt"):
        path = run_dir / candidate
        if path.exists():
            return path
    return None


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)
    preexisting_runs = {p.name for p in models_dir.iterdir() if p.is_dir()}

    train_env = os.environ.copy()
    if not args.use_wandb and "WANDB_MODE" not in train_env:
        train_env["WANDB_MODE"] = "disabled"

    print(f"[pipeline] Starting training: {args.train_command}")
    train_cmd = shlex.split(args.train_command)
    train_rc, train_output = run_command(train_cmd, env=train_env)

    run_id = parse_run_id(train_output, preexisting_runs, models_dir)
    print(f"[pipeline] Training exit code: {train_rc}")
    print(f"[pipeline] Parsed run_id: {run_id or 'unknown'}")

    if run_id is None:
        print("[pipeline] Could not detect run_id. Stopping before verifier.")
        return 1

    run_dir = models_dir / run_id
    checkpoint = resolve_checkpoint(run_dir)
    if checkpoint is None:
        print(f"[pipeline] No checkpoint found in {run_dir}.")
        return 1

    config = load_defaults_from_main(repo_root / "main.py")
    if args.config_overrides_json:
        with open(args.config_overrides_json, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        deep_update(config, overrides)

    run_config_path = run_dir / "pipeline_config.json"
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(
        "[pipeline] Config snapshot:",
        json.dumps(
            {
                "dim": config.get("dim"),
                "n_colours": config.get("n_colours"),
                "parallelogram": config.get("training", {}).get("parallelogram"),
                "model": config.get("model", {}).get("name"),
            }
        ),
    )

    verify_cmd = [
        sys.executable,
        "scripts/verify_paralellogram_ip.py",
        "--run_id",
        run_id,
        "--eval_gridsize",
        str(args.eval_gridsize),
        "--checkpoint_path",
        str(checkpoint),
        "--config_json",
        str(run_config_path),
    ]
    if args.use_wandb:
        verify_cmd += ["--wandb_project", args.wandb_project, "--wandb_entity", args.wandb_entity]

    verify_env = os.environ.copy()
    print(f"[pipeline] Starting ILP verification at {args.eval_gridsize}x{args.eval_gridsize}.")
    verify_rc, verify_output = run_command(verify_cmd, env=verify_env)
    print(f"[pipeline] Verification exit code: {verify_rc}")

    bonus_match = BONUS_FRAC_RE.search(verify_output)
    if bonus_match:
        print(f"[pipeline] Final verified bonus color percentage: {bonus_match.group(1)}%")
    else:
        print("[pipeline] Bonus percentage not found in verifier output.")

    if train_rc != 0:
        print("[pipeline] Training failed, but partial pipeline output is printed above.")
    if verify_rc != 0:
        print("[pipeline] Verification failed, but partial pipeline output is printed above.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
