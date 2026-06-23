#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def extract_jsonish(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, flags=re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def repair_json_escapes(text: str) -> str:
    # Agent-authored command strings often contain shell escapes such as \',
    # which are invalid JSON escapes. Preserve valid JSON escapes and double
    # everything else so the text remains literal inside strings.
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def read_jsonish_file(path: Path) -> dict:
    raw = path.read_text(errors="replace")
    candidate = extract_jsonish(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        data = json.loads(repair_json_escapes(candidate))
    if not isinstance(data, dict):
        raise SystemExit("JSON artifact must be an object: " + str(path))
    return data


def markdown_value(value: object, depth: int = 0) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            label = str(key).replace("_", " ")
            rendered = markdown_value(item, depth + 1)
            if "\n" in rendered:
                lines.append("- " + label + ":")
                lines.extend("  " + line if line else "" for line in rendered.splitlines())
            else:
                lines.append("- " + label + ": " + rendered)
        return "\n".join(lines).strip()
    if isinstance(value, list):
        lines = []
        for item in value:
            rendered = markdown_value(item, depth + 1)
            if "\n" in rendered:
                lines.append("-")
                lines.extend("  " + line if line else "" for line in rendered.splitlines())
            else:
                lines.append("- " + rendered)
        return "\n".join(lines).strip()
    return str(value).strip()


def normalize_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def run(args: list[str], cwd: Path, allow_failure: bool = False) -> str:
    proc = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0 and not allow_failure:
        raise SystemExit("command failed: " + " ".join(args) + "\n" + proc.stdout)
    return proc.stdout


def require_shared_python(root: Path) -> Path:
    shared = Path(os.environ.get("AAC_SHARED_VENV", "") or root / "shared_venv")
    py = shared / "bin" / "python"
    if not py.exists():
        raise SystemExit("missing shared venv python: " + str(py))
    return py


ENSEMBLE_EVAL = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from models import ResMLP
from utilities import GeneralUtility


def evaluate_run_dir(run_dir, eval_grid_size=128, eval_circle_points=128, device=None):
    run_dir = Path(run_dir)
    config_path = run_dir / "pipeline_config.json"
    checkpoint_path = run_dir / "trained_model.pt"
    if not config_path.exists():
        raise FileNotFoundError("missing pipeline_config.json: " + str(config_path))
    if not checkpoint_path.exists():
        raise FileNotFoundError("missing trained_model.pt: " + str(checkpoint_path))
    config = json.loads(config_path.read_text())
    dev = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_kwargs = dict(config["model"])
    model_kwargs.pop("name", None)
    model = ResMLP(input_dim=int(config["dim"]), output_dim=int(config["n_colours"]), device=dev, **model_kwargs)
    state = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(state)
    model = model.to(dev).eval()
    parallelogram = torch.tensor(config["training"]["parallelogram"], device=dev, dtype=torch.float32)
    wrapped = GeneralUtility.prepend_parallelogram_transformation(model, spanning_vectors=parallelogram)
    metrics = GeneralUtility.get_parallelogram_coloring_metrics(
        model=wrapped,
        parallelogram=parallelogram,
        gridsize=int(eval_grid_size),
        n_circle_points=int(eval_circle_points),
        n_colours=int(config["n_colours"]),
    )
    return {
        "bonus_fraction_pct": 100.0 * float(metrics["bonus_fraction"]),
        "real_conflict_fraction_pct": 100.0 * float(metrics["real_conflict_fraction"]),
        "bad_fraction_pct": 100.0 * float(metrics["bad_fraction"]),
        "eval_grid_size": int(eval_grid_size),
        "eval_circle_points": int(eval_circle_points),
    }
"""


def cmd_preflight(args: argparse.Namespace) -> None:
    repo = Path(args.repo_root).resolve()
    root = Path(args.root).resolve()
    source = Path(args.source_worktree).resolve()
    rounds = int(args.rounds)
    calibration_seconds = int(args.calibration_seconds)
    agent_timeout_seconds = int(args.agent_timeout_seconds)
    target_pct = float(args.target_pct)
    baseline_pct = float(args.baseline_pct)
    if not 1 <= rounds <= 20:
        raise SystemExit("rounds must be in [1,20], got " + str(rounds))
    if not 60 <= calibration_seconds <= 1800:
        raise SystemExit("calibrationSeconds must be in [60,1800], got " + str(calibration_seconds))
    if not 300 <= agent_timeout_seconds <= 7200:
        raise SystemExit("agentTimeoutSeconds must be in [300,7200], got " + str(agent_timeout_seconds))

    required = [
        "scripts/train_batched_ensemble.py",
        "scripts/summarize_training_sweep.py",
        "scripts/verify_paralellogram_ip.py",
        "scripts/verify_parallelogram_heuristic.py",
        "models.py",
        "utilities.py",
        "pyproject.toml",
    ]
    missing = [path for path in required if not (source / path).exists()]
    if missing:
        raise SystemExit("base worktree missing required files: " + repr(missing))

    helper_path = source / "scripts" / "ensemble_eval.py"
    helper_was_missing = not helper_path.exists()
    helper_path.write_text(ENSEMBLE_EVAL)
    helper_path.chmod(0o755)
    (root / "state" / "ensemble_eval.py").write_text(ENSEMBLE_EVAL)

    smoke_root = root / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["MPLCONFIGDIR"] = str(root / "mplconfig")
    env["UV_NO_SYNC"] = "1"
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    py = require_shared_python(root)
    command = [
        str(py),
        "scripts/train_batched_ensemble.py",
        "--ensemble-size",
        "1",
        "--n-steps",
        "2",
        "--batch-size",
        "64",
        "--n-circle-points",
        "2",
        "--loss-log-every",
        "2",
        "--skip-eval",
        "--output-root",
        str(smoke_root),
        "--sweep-id",
        "smoke_train_only",
        "--base-seed",
        "9100000",
    ]
    proc = subprocess.run(command, cwd=source, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (root / "state" / "smoke_train_only.log").write_text(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit("train_batched_ensemble smoke failed; see " + str(root / "state" / "smoke_train_only.log"))

    inventory = {
        "workflow": "batched-neural-repair-verification-loop",
        "repo_root": str(repo),
        "base_ref": args.base_ref,
        "base_head": args.base_head,
        "source_worktree": str(source),
        "output_root": str(root),
        "shared_venv": os.environ["AAC_SHARED_VENV"],
        "helper_was_missing": helper_was_missing,
        "helper_path": str(helper_path),
        "rounds": rounds,
        "calibration_seconds": calibration_seconds,
        "agent_timeout_seconds": agent_timeout_seconds,
        "target_pct": target_pct,
        "baseline_pct": baseline_pct,
        "created_at": time.time(),
    }
    write_json(root / "state" / "inventory.json", inventory)
    brief = [
        "# Batched Neural Repair Verification Loop",
        "",
        "Target: find a verified almost-5-coloring below " + str(target_pct) + " percent bonus cells.",
        "Baseline for promotion: " + str(baseline_pct) + " percent, with zero real-color conflicts required.",
        "",
        "Base ref: `" + args.base_ref + "` at `" + args.base_head + "`.",
        "Source worktree: `" + str(source) + "`.",
        "Shared venv: `" + os.environ["AAC_SHARED_VENV"] + "`.",
        "",
        "Campaign structure:",
        "- deterministic integration and smoke gate;",
        "- five-minute one-GPU ensemble-capacity calibration;",
        "- 1-20 reflection rounds with four parallel agents and a shared GPU flock lock;",
        "- promotion ladder: 128-grid proxy bad_fraction_pct, 256/512 proxy, heuristic repair, component MILP, independent verification;",
        "- final fanout recommendation for many seeds/parallelogram jitters aimed at below 3.60 percent.",
        "",
        "Important branch issue: scripts/ensemble_eval.py was missing from the fetched branch, so this workflow installs a minimal helper inside scratch worktrees only.",
    ]
    (root / "design_brief.md").write_text("\n".join(brief) + "\n")
    print(json.dumps(inventory, indent=2, sort_keys=True))


def cmd_capacity(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    inventory = json.loads((root / "state" / "inventory.json").read_text())
    source = Path(inventory["source_worktree"])
    py = require_shared_python(root)
    calibration_seconds = int(args.calibration_seconds)
    output_root = root / "capacity_calibration"
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["MPLCONFIGDIR"] = str(root / "mplconfig")
    env["UV_NO_SYNC"] = "1"
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    tests = []
    for size in [16, 32, 64, 96, 128, 192, 256]:
        tests.append({"ensemble_size": size, "layers": 2, "units": 64, "steps": 180})
    for size in [64, 96, 128, 192]:
        tests.append({"ensemble_size": size, "layers": 4, "units": 32, "steps": 180})
    for size in [32, 64, 96]:
        tests.append({"ensemble_size": size, "layers": 3, "units": 64, "steps": 180})

    deadline = time.time() + calibration_seconds
    records = []
    for idx, test in enumerate(tests):
        remaining = deadline - time.time()
        if remaining < 35:
            break
        sweep_id = "capacity_" + str(idx).zfill(2) + "_e" + str(test["ensemble_size"]) + "_l" + str(test["layers"]) + "_u" + str(test["units"])
        command = [
            str(py),
            "scripts/train_batched_ensemble.py",
            "--ensemble-size",
            str(test["ensemble_size"]),
            "--n-steps",
            str(test["steps"]),
            "--batch-size",
            "2048",
            "--n-circle-points",
            "8",
            "--loss-log-every",
            str(test["steps"]),
            "--skip-eval",
            "--output-root",
            str(output_root),
            "--sweep-id",
            sweep_id,
            "--base-seed",
            str(9200000 + idx * 10000),
            "--n-hidden-layers",
            str(test["layers"]),
            "--n-hidden-units",
            str(test["units"]),
            "--activation",
            "sin",
            "--initialization",
            "siren",
        ]
        start = time.time()
        timed_out = False
        try:
            proc = subprocess.run(
                command,
                cwd=source,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=max(20, min(65, int(remaining - 5))),
            )
            rc = proc.returncode
            out = proc.stdout
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            rc = 124
            out = exc.stdout or ""
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            out += "\n[TIMEOUT]\n"
        elapsed = time.time() - start
        log_path = log_dir / (sweep_id + ".log")
        log_path.write_text(out)
        lower = out.lower()
        record = dict(test)
        record.update(
            {
                "sweep_id": sweep_id,
                "returncode": rc,
                "success": rc == 0,
                "timed_out": timed_out,
                "elapsed_s": elapsed,
                "step_time_s": elapsed / max(1, test["steps"]),
                "model_step_per_s": (test["ensemble_size"] * test["steps"] / elapsed) if elapsed > 0 else 0.0,
                "oom": "out of memory" in lower or "cuda error" in lower,
                "log_path": str(log_path),
            }
        )
        records.append(record)

    successful = [r for r in records if r["success"]]
    best = max(successful, key=lambda r: (r["model_step_per_s"], r["ensemble_size"])) if successful else {
        "ensemble_size": 16,
        "layers": 2,
        "units": 64,
        "model_step_per_s": 0.0,
        "step_time_s": None,
    }
    summary = {
        "records": records,
        "recommended_ensemble_size": int(best["ensemble_size"]),
        "recommended_layers": int(best["layers"]),
        "recommended_units": int(best["units"]),
        "recommended_model_step_per_s": float(best["model_step_per_s"]),
        "recommended_step_time_s": best["step_time_s"],
        "calibration_seconds": calibration_seconds,
        "output_root": str(output_root),
    }
    write_json(root / "state" / "capacity_calibration.json", summary)
    lines = [
        "# Capacity calibration",
        "",
        "Recommended ensemble size: " + str(summary["recommended_ensemble_size"]),
        "Recommended architecture: " + str(summary["recommended_layers"]) + " layers x " + str(summary["recommended_units"]) + " units",
        "",
        "| ensemble | layers | units | success | elapsed_s | step_time_s | model_step_per_s | oom |",
        "| ---: | ---: | ---: | :---: | ---: | ---: | ---: | :---: |",
    ]
    for r in records:
        lines.append(
            "| "
            + str(r["ensemble_size"])
            + " | "
            + str(r["layers"])
            + " | "
            + str(r["units"])
            + " | "
            + str(r["success"])
            + " | "
            + format(r["elapsed_s"], ".2f")
            + " | "
            + format(r["step_time_s"], ".4f")
            + " | "
            + format(r["model_step_per_s"], ".2f")
            + " | "
            + str(r["oom"])
            + " |"
        )
    (root / "state" / "capacity_calibration.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_validate_seed(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    path = root / "state" / "seed_instructions.json"
    if not path.exists():
        raise SystemExit("seed instruction file missing: " + str(path))
    data = read_jsonish_file(path)
    required = ["round_summary", "best_findings", "rejected_or_weak_findings", "risks", "agent_1", "agent_2", "agent_3", "agent_4"]
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit("seed JSON missing required keys " + str(missing) + "; got " + str(sorted(data.keys())))
    normalized = {key: data[key] for key in required}
    for key in ["best_findings", "rejected_or_weak_findings", "risks"]:
        normalized[key] = normalize_list(normalized.get(key))
    for key in ["agent_1", "agent_2", "agent_3", "agent_4"]:
        normalized[key] = markdown_value(normalized.get(key))
        if not normalized[key]:
            raise SystemExit(key + " must be non-empty after normalization")
    write_json(path, normalized)
    cur = root / "state" / "current_instructions"
    cur.mkdir(parents=True, exist_ok=True)
    for k in range(1, 5):
        key = "agent_" + str(k)
        (cur / (key + ".md")).write_text(normalized[key].strip() + "\n")
    shutil.copyfile(path, root / "state" / "previous_reflection.json")
    print(json.dumps({"seed_instructions": str(path), "current_instruction_dir": str(cur)}, indent=2, sort_keys=True))


def cmd_round_preflight(args: argparse.Namespace) -> None:
    repo = Path(args.repo_root).resolve()
    root = Path(args.root).resolve()
    r = int(args.iteration)
    round_name = "round_" + str(r)
    round_dir = root / round_name
    round_dir.mkdir(parents=True, exist_ok=True)
    (root / "worktrees" / round_name).mkdir(parents=True, exist_ok=True)
    (root / "jobs" / round_name).mkdir(parents=True, exist_ok=True)
    inv = json.loads((root / "state" / "inventory.json").read_text())
    cal = json.loads((root / "state" / "capacity_calibration.json").read_text())
    base = inv["base_head"]
    cur = root / "state" / "current_instructions"
    helper_src = root / "state" / "ensemble_eval.py"
    gpu_lock = root / "gpu.lock"
    gpu_lock.touch(exist_ok=True)
    run_slug_base = re.sub(r"[^A-Za-z0-9._-]+", "-", root.name).strip("-") or "run"
    run_slug_hash = hashlib.sha1(str(root).encode()).hexdigest()[:8]
    run_slug = (run_slug_base[:36] + "-" + run_slug_hash).strip("-")
    shared_venv = os.environ.get("AAC_SHARED_VENV", "").strip()
    shared_python = str(Path(shared_venv) / "bin" / "python") if shared_venv else ""
    worktrees = []
    branches = []
    for k in range(1, 5):
        agent_name = "agent_" + str(k)
        agent_dir = round_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        wt = root / "worktrees" / round_name / agent_name
        branch = "wf/batched-neural/" + run_slug + "/round-" + str(r) + "/agent-" + str(k)
        run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo, allow_failure=True)
        if wt.exists():
            shutil.rmtree(wt, ignore_errors=True)
        run(["git", "worktree", "prune"], cwd=repo, allow_failure=True)
        run(["git", "branch", "-D", branch], cwd=repo, allow_failure=True)
        run(["git", "worktree", "add", "-b", branch, str(wt), base], cwd=repo)
        helper_dst = wt / "scripts" / "ensemble_eval.py"
        if helper_src.exists():
            helper_dst.write_text(helper_src.read_text())
            helper_dst.chmod(0o755)
        instr_text = (cur / (agent_name + ".md")).read_text().strip()
        (agent_dir / "instruction.md").write_text(instr_text + "\n")
        prompt_lines = [
            "You are explorer agent " + str(k) + " for round " + str(r) + " of the batched-neural-repair-verification-loop workflow.",
            "",
            "HARD SCOPE RULES:",
            "- Work only inside your assigned git worktree for repository modifications: " + str(wt),
            "- Required report path: " + str(agent_dir / "report.md"),
            "- Optional artifacts manifest path: " + str(agent_dir / "artifacts.json"),
            "- Use scratch output directories under: " + str(agent_dir / "experiments"),
            "- All GPU training/evaluation commands must use the shared lock so four agents do not fight over one GPU: flock " + str(gpu_lock) + " COMMAND ...",
            "- Shared Python interpreter: " + shared_python,
            "- If you use uv, use UV_PROJECT_ENVIRONMENT=" + shared_venv + " uv run --frozen --no-sync ... ; do not run uv sync or create .venv directories.",
            "- No-improvement is useful evidence. Report it clearly.",
            "",
            "CAMPAIGN STATE:",
            "- Design brief: " + str(root / "design_brief.md"),
            "- Inventory: " + str(root / "state" / "inventory.json"),
            "- Capacity calibration JSON: " + str(root / "state" / "capacity_calibration.json"),
            "- Capacity calibration markdown: " + str(root / "state" / "capacity_calibration.md"),
            "- Previous reflection: " + str(root / "state" / "previous_reflection.json"),
            "- Prior sibling summaries: " + str(root / "state" / "prior_sibling_summaries.md"),
            "- Target verified bonus percent: " + str(args.target_pct),
            "- Promotion baseline percent: " + str(args.baseline_pct),
            "- Recommended ensemble size: " + str(cal.get("recommended_ensemble_size")),
            "- Recommended architecture from calibration: " + str(cal.get("recommended_layers")) + "x" + str(cal.get("recommended_units")),
            "",
            "PROMOTION LADDER:",
            "1. cheap proxy: train_batched_ensemble summary.csv sorted by bad_fraction_pct at 128 grid;",
            "2. stronger proxy: re-evaluate top candidates at 256 or 512 grid;",
            "3. repair: scripts/verify_parallelogram_heuristic.py with MIS, augment, and kick variants;",
            "4. certificate: scripts/verify_paralellogram_ip.py component MILP or CP-SAT with zero real-color conflicts;",
            "5. final: independent verification artifact and exact bonus fraction below target.",
            "",
            "YOUR CURRENT INSTRUCTION:",
            instr_text,
            "",
            "REPORT REQUIREMENTS:",
            "- Status: completed | no_improvement | partial | blocked",
            "- Experiments run, exact commands, paths, elapsed time, and whether GPU lock was used",
            "- Best metric rows found and why they matter",
            "- Failed/weak ideas and negative evidence",
            "- Concrete next instruction for a future agent",
            "- Artifact paths worth preserving",
        ]
        (agent_dir / "prompt.md").write_text("\n".join(prompt_lines) + "\n")
        worktrees.append(str(wt))
        branches.append(branch)
    print(json.dumps({"round": r, "round_dir": str(round_dir), "worktrees": worktrees, "branches": branches, "gpu_lock": str(gpu_lock)}, indent=2, sort_keys=True))


def cmd_collect_round(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    r = int(args.iteration)
    round_dir = root / ("round_" + str(r))
    db = root / "jobs" / "batched-neural-repair-verification-loop.sqlite"
    if not db.exists():
        raise SystemExit("async jobs DB missing after explorer phase: " + str(db))
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    pattern = "batched-neural-r" + str(r) + "-a%"
    rows = {row["id"]: dict(row) for row in con.execute("select * from workflow_jobs where id like ? order by id", (pattern,))}
    con.close()
    summary_lines = ["# Round " + str(r) + " explorer outputs", "", "Jobs DB: `" + str(db) + "`", ""]
    status = {"round": r, "agents": {}}
    prior_append = []
    for k in range(1, 5):
        agent_name = "agent_" + str(k)
        agent_dir = round_dir / agent_name
        report = agent_dir / "report.md"
        artifacts = agent_dir / "artifacts.json"
        job_id = "batched-neural-r" + str(r) + "-a" + str(k)
        job = rows.get(job_id)
        if job is None:
            raise SystemExit("missing async job row for " + job_id)
        stdout_path = Path(job["stdout_path"])
        stderr_path = Path(job["stderr_path"])
        stdout_tail = stdout_path.read_text(errors="replace")[-6000:] if stdout_path.exists() else ""
        stderr_tail = stderr_path.read_text(errors="replace")[-6000:] if stderr_path.exists() else ""
        report_text = report.read_text(errors="replace") if report.exists() else None
        artifacts_text = None
        if artifacts.exists():
            artifacts_text = artifacts.read_text(errors="replace")
            json.loads(artifacts_text)
        status["agents"][agent_name] = {
            "job_id": job_id,
            "job_status": job.get("status"),
            "exit_code": job.get("exit_code"),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "report_path": str(report),
            "report_exists": report.exists(),
            "artifacts_path": str(artifacts) if artifacts.exists() else None,
            "timed_out_or_failed_is_search_result": job.get("status") != "succeeded",
        }
        summary_lines.extend(
            [
                "## " + agent_name,
                "",
                "- job_id: `" + job_id + "`",
                "- job_status: `" + str(job.get("status")) + "`",
                "- exit_code: `" + str(job.get("exit_code")) + "`",
                "- report: `" + str(report) + "` (" + ("present" if report.exists() else "missing") + ")",
                "",
            ]
        )
        if report_text:
            clipped = report_text if len(report_text) <= 24000 else report_text[:24000] + "\n...[report clipped for reflection input]"
            summary_lines.extend(["### report.md", "", clipped, ""])
            prior_append.extend(["## Round " + str(r) + " " + agent_name, "", clipped[:8000], ""])
        else:
            summary_lines.extend(
                [
                    "### No report was produced",
                    "This is treated as timeout/blocked search evidence unless logs are unreadable.",
                    "",
                    "#### stdout tail",
                    "```",
                    stdout_tail,
                    "```",
                    "#### stderr tail",
                    "```",
                    stderr_tail,
                    "```",
                    "",
                ]
            )
        if artifacts_text:
            summary_lines.extend(["### artifacts.json", "```json", artifacts_text[:12000], "```", ""])
    write_json(round_dir / "round_status.json", status)
    (round_dir / "round_summary_input.md").write_text("\n".join(summary_lines))
    prior_path = root / "state" / "prior_sibling_summaries.md"
    with prior_path.open("a") as f:
        f.write("\n".join(prior_append))
        if prior_append:
            f.write("\n")
    print(json.dumps({"round": r, "round_summary_input": str(round_dir / "round_summary_input.md"), "round_status": str(round_dir / "round_status.json")}, indent=2, sort_keys=True))


def cmd_validate_reflection(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    r = int(args.iteration)
    path = root / ("round_" + str(r)) / "reflection.json"
    if not path.exists():
        raise SystemExit("reflection JSON missing: " + str(path))
    data = read_jsonish_file(path)
    required = ["round_summary", "best_findings", "rejected_or_weak_findings", "risks", "runner_plan", "agent_1", "agent_2", "agent_3", "agent_4"]
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit("reflection JSON missing required keys " + str(missing) + "; got " + str(sorted(data.keys())))
    normalized = {key: data[key] for key in required}
    for key in ["round_summary", "runner_plan", "agent_1", "agent_2", "agent_3", "agent_4"]:
        normalized[key] = markdown_value(normalized.get(key))
        if not normalized[key]:
            raise SystemExit(key + " must be non-empty after normalization")
    for key in ["best_findings", "rejected_or_weak_findings", "risks"]:
        normalized[key] = normalize_list(normalized.get(key))
    write_json(path, normalized)
    cur = root / "state" / "current_instructions"
    cur.mkdir(parents=True, exist_ok=True)
    for k in range(1, 5):
        key = "agent_" + str(k)
        (cur / (key + ".md")).write_text(normalized[key].strip() + "\n")
    shutil.copyfile(path, root / "state" / "previous_reflection.json")
    print(json.dumps({"round": r, "reflection": str(path), "next_instruction_dir": str(cur)}, indent=2, sort_keys=True))


def cmd_validate_final(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    final = root / "final_synthesis.md"
    fanout = root / "final_fanout_plan.json"
    if not final.exists():
        raise SystemExit("final synthesis missing: " + str(final))
    if len(final.read_text(errors="replace").strip()) < 200:
        raise SystemExit("final synthesis too short: " + str(final))
    if not fanout.exists():
        raise SystemExit("final fanout plan missing: " + str(fanout))
    json.loads(fanout.read_text())
    print(json.dumps({"final_synthesis": str(final), "final_fanout_plan": str(fanout), "output_root": str(root), "jobs_db": str(root / "jobs" / "batched-neural-repair-verification-loop.sqlite")}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--repo-root", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--source-worktree", required=True)
    p.add_argument("--base-ref", required=True)
    p.add_argument("--base-head", required=True)
    p.add_argument("--rounds", required=True)
    p.add_argument("--calibration-seconds", required=True)
    p.add_argument("--agent-timeout-seconds", required=True)
    p.add_argument("--target-pct", required=True)
    p.add_argument("--baseline-pct", required=True)
    p.set_defaults(func=cmd_preflight)
    p = sub.add_parser("capacity")
    p.add_argument("--root", required=True)
    p.add_argument("--calibration-seconds", required=True)
    p.set_defaults(func=cmd_capacity)
    p = sub.add_parser("validate-seed")
    p.add_argument("--root", required=True)
    p.set_defaults(func=cmd_validate_seed)
    p = sub.add_parser("round-preflight")
    p.add_argument("--repo-root", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--iteration", required=True)
    p.add_argument("--target-pct", required=True)
    p.add_argument("--baseline-pct", required=True)
    p.set_defaults(func=cmd_round_preflight)
    p = sub.add_parser("collect-round")
    p.add_argument("--root", required=True)
    p.add_argument("--iteration", required=True)
    p.set_defaults(func=cmd_collect_round)
    p = sub.add_parser("validate-reflection")
    p.add_argument("--root", required=True)
    p.add_argument("--iteration", required=True)
    p.set_defaults(func=cmd_validate_reflection)
    p = sub.add_parser("validate-final")
    p.add_argument("--root", required=True)
    p.set_defaults(func=cmd_validate_final)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
