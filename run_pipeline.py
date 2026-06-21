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


RUN_DIR_RE = re.compile(r"Saved persistent local copy of all outputs to (.+)\.")
BONUS_FRAC_RE = re.compile(r"Fraction set to bonus color:\s*([0-9.]+)%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a model and automatically run ILP verification."
    )
    parser.add_argument(
        "--train-command",
        default="python main.py --debug --fast-train",
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
    parser.add_argument(
        "--verify-output-root",
        default=None,
        help="Optional root directory for verifier artifacts. Defaults to the run directory.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip final verifier plot generation.",
    )
    parser.add_argument(
        "--solver-time-limit",
        type=int,
        default=360000,
        help="CBC solver time limit in seconds.",
    )
    parser.add_argument(
        "--solver-threads",
        type=int,
        default=1,
        help="Number of CBC solver threads for verifier MILPs.",
    )
    parser.add_argument(
        "--solver-backend",
        choices=("cbc", "scip", "cp_sat"),
        default="cbc",
        help="Verifier component solver backend.",
    )
    parser.add_argument(
        "--solver-gap-abs",
        type=float,
        default=None,
        help="Absolute MIP optimality gap for verifier solvers.",
    )
    parser.add_argument(
        "--solver-gap-rel",
        type=float,
        default=None,
        help="Relative MIP optimality gap for verifier solvers.",
    )
    parser.add_argument(
        "--solver-max-nodes",
        type=int,
        default=None,
        help="Maximum branch-and-bound nodes for verifier MIP solvers.",
    )
    parser.add_argument(
        "--cbc-cuts",
        choices=("default", "on", "off"),
        default="default",
        help="CBC cut generation setting.",
    )
    parser.add_argument(
        "--cbc-presolve",
        choices=("default", "on", "off"),
        default="default",
        help="CBC presolve setting.",
    )
    parser.add_argument(
        "--cbc-strong",
        type=int,
        default=None,
        help="CBC strong branching candidate count.",
    )
    parser.add_argument(
        "--scip-path",
        default=os.getenv("SCIP_PATH", "/software/opt-sw/scipoptsuite-10.0.2/bin/scip"),
        help="Path to SCIP executable when --solver-backend=scip.",
    )
    parser.add_argument(
        "--neighbor-backend",
        choices=("fft", "conv"),
        default="fft",
        help="Verifier neighbor-query backend.",
    )
    parser.add_argument(
        "--mask-cache-dir",
        default=None,
        help="Optional shared verifier base-mask cache directory.",
    )
    parser.add_argument(
        "--skip-vertex-cover",
        action="store_true",
        help="Let the MILP handle all remaining conflicted cells instead of using the vertex-cover pre-repair.",
    )
    parser.add_argument(
        "--active-edge-chunk-size",
        type=int,
        default=64,
        help="Verifier mask-offset chunk size for vectorized active-edge discovery.",
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


def parse_run_dir(training_output: str, preexisting_runs: set[str], models_dir: Path) -> Optional[Path]:
    match = RUN_DIR_RE.search(training_output)
    if match:
        return Path(match.group(1).strip())

    current_runs = {p.name for p in models_dir.iterdir() if p.is_dir()} if models_dir.exists() else set()
    new_runs = sorted(current_runs - preexisting_runs)
    if new_runs:
        newest = max((models_dir / run_id for run_id in new_runs), key=lambda p: p.stat().st_mtime)
        return newest

    if current_runs:
        newest = max((models_dir / run_id for run_id in current_runs), key=lambda p: p.stat().st_mtime)
        return newest
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

    run_dir = parse_run_dir(train_output, preexisting_runs, models_dir)
    run_id = run_dir.name if run_dir is not None else None
    print(f"[pipeline] Training exit code: {train_rc}")
    print(f"[pipeline] Parsed run_id: {run_id or 'unknown'}")

    if run_dir is None or run_id is None:
        print("[pipeline] Could not detect run_id. Stopping before verifier.")
        return 1

    checkpoint = resolve_checkpoint(run_dir)
    if checkpoint is None:
        print(f"[pipeline] No checkpoint found in {run_dir}.")
        return 1

    run_config_path = run_dir / "pipeline_config.json"
    if run_config_path.exists():
        with open(run_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = load_defaults_from_main(repo_root / "main.py")

    if args.config_overrides_json:
        with open(args.config_overrides_json, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        deep_update(config, overrides)

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

    if args.verify_output_root:
        verify_output_dir = Path(args.verify_output_root) / f"{run_id}_grid{args.eval_gridsize}"
    else:
        verify_output_dir = run_dir / f"verification_grid{args.eval_gridsize}"

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
        "--output_dir",
        str(verify_output_dir),
        "--solver_time_limit",
        str(args.solver_time_limit),
        "--solver_threads",
        str(args.solver_threads),
        "--solver_backend",
        args.solver_backend,
        "--scip_path",
        args.scip_path,
        "--cbc_cuts",
        args.cbc_cuts,
        "--cbc_presolve",
        args.cbc_presolve,
        "--neighbor_backend",
        args.neighbor_backend,
        "--active_edge_chunk_size",
        str(args.active_edge_chunk_size),
    ]
    if args.solver_gap_abs is not None:
        verify_cmd += ["--solver_gap_abs", str(args.solver_gap_abs)]
    if args.solver_gap_rel is not None:
        verify_cmd += ["--solver_gap_rel", str(args.solver_gap_rel)]
    if args.solver_max_nodes is not None:
        verify_cmd += ["--solver_max_nodes", str(args.solver_max_nodes)]
    if args.cbc_strong is not None:
        verify_cmd += ["--cbc_strong", str(args.cbc_strong)]
    if args.mask_cache_dir:
        verify_cmd += ["--mask_cache_dir", args.mask_cache_dir]
    if args.skip_vertex_cover:
        verify_cmd.append("--skip_vertex_cover")
    if args.no_plot:
        verify_cmd.append("--no_plot")
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
