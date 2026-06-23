#!/usr/bin/env python3
"""Deterministic verifier-only coarse-to-fine parallelogram search helper.

This script is intentionally workflow-owned: it generates candidates, runs verifier
subprocesses with scratch tmp dirs, promotes by measured feasible score, and writes
concise markdown reports. Lack of improvement is recorded but is not an error.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REQUIRED = [
    Path("scripts/verify_paralellogram_ip.py"),
    Path("scripts/verify_paralellogram_ip_rect.py"),
    Path("scripts/verify_parallelogram_heuristic.py"),
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys or ["empty"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if rows:
            w.writerows(rows)


def load_json(path: Path):
    return json.loads(path.read_text())


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def pct_from_verification_csv(path: Path) -> float | None:
    rows = read_csv(path)
    if not rows:
        return None
    for key in ("fraction_fixed_to_5 (%)", "score", "final_pct", "screen_pct"):
        val = rows[-1].get(key)
        if val not in (None, ""):
            try:
                return float(val)
            except Exception:
                pass
    return None


def discover_seed(root: Path, baseline: float) -> dict:
    seeds: list[dict] = []

    def add(run_id, ckpt, cfg, source, pct=None):
        if not ckpt or not cfg:
            return
        cp, cf = Path(ckpt), Path(cfg)
        if cp.exists() and cf.exists():
            seeds.append({
                "run_id": str(run_id or cp.parent.name),
                "checkpoint_path": str(cp),
                "config_json": str(cf),
                "source": str(source),
                "known_pct": None if pct in (None, "") else float(pct),
            })

    for cfg in list(Path(".").glob("**/pipeline_config.json")) + list(Path(".").glob("**/hparams.json")):
        if ".venv" in cfg.parts or ".workflow_runs" in cfg.parts:
            continue
        ckpt = cfg.with_name("trained_model.pt")
        pct = pct_from_verification_csv(cfg.parent.parent / "verification" / "verified_paralellograms_ip.csv")
        add(cfg.parent.name, ckpt, cfg, "co-located-config-checkpoint", pct)

    for csv_path in [Path("slurm/batched32_hparam_1247042_top10.csv"), Path(".workflow_runs/parallelogram-improvement-search/global_results.csv")]:
        for row in read_csv(csv_path):
            add(row.get("run_id"), row.get("checkpoint_path"), row.get("config_json"), str(csv_path), row.get("screen_pct"))

    for row in read_csv(Path("slurm/top_rect_2048_candidates.csv")):
        rd = row.get("run_dir")
        if rd:
            add(row.get("run_id"), Path(rd) / "trained_model.pt", Path(rd) / "pipeline_config.json", "slurm/top_rect_2048_candidates.csv", row.get("score"))

    uniq = {}
    for s in seeds:
        key = (s["checkpoint_path"], s["config_json"])
        old = uniq.get(key)
        if old is None or ((s.get("known_pct") is not None) and (old.get("known_pct") is None or s["known_pct"] < old["known_pct"])):
            uniq[key] = s
    seeds = sorted(uniq.values(), key=lambda s: (s["known_pct"] if s.get("known_pct") is not None else 999.0, s["run_id"]))
    if not seeds:
        raise SystemExit("no checkpoint/config seed pairs found")
    seed = seeds[0]
    base_cfg = load_json(Path(seed["config_json"]))
    base_para = base_cfg["training"]["parallelogram"]
    state = {
        "baselinePct": baseline,
        "bestPct": seed.get("known_pct", baseline) or baseline,
        "bestParallelogram": base_para,
        "bestSource": "initial-seed",
        "seed": seed,
        "history": [],
    }
    dump_json(root / "inventory.json", {"required_scripts": [str(p) for p in REQUIRED], "seed_count": len(seeds), "seed": seed, "seeds": seeds[:50]})
    dump_json(root / "state.json", state)
    return state


def det2(p):
    return float(p[0][0]) * float(p[1][1]) - float(p[0][1]) * float(p[1][0])


def norm(v):
    return math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2)


def ok_geometry(p, base, area_tol: float, norm_tol: float, min_det: float) -> tuple[bool, str]:
    d, bd = abs(det2(p)), abs(det2(base))
    if d < min_det:
        return False, "det-too-small"
    if bd > 0 and abs(d / bd - 1.0) > area_tol:
        return False, "area-ratio"
    for i in range(2):
        n, bn = norm(p[i]), norm(base[i])
        if bn > 0 and abs(n / bn - 1.0) > norm_tol:
            return False, "norm-ratio"
    return True, "ok"


def load_state(root: Path) -> dict:
    return load_json(root / "state.json")


def make_candidates(args) -> None:
    root = Path(args.root)
    state = load_state(root)
    seed = state["seed"]
    base_cfg = load_json(Path(seed["config_json"]))
    base_para = base_cfg["training"]["parallelogram"]
    center = state.get("bestParallelogram") or base_para
    radius = float(args.base_radius) / (2 ** max(0, int(args.iteration) - 1))
    rng = random.Random(f"{args.iteration}:{json.dumps(center)}")
    rows = []
    out_dir = root / "iterations" / f"iter_{int(args.iteration)}" / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    def add_candidate(family, label, p):
        ok, reason = ok_geometry(p, base_para, float(args.area_tol), float(args.norm_tol), float(args.min_det))
        if not ok:
            return
        cid = f"i{int(args.iteration):03d}_{len(rows):05d}_{family}_{label}"
        cfg = copy.deepcopy(base_cfg)
        cfg["training"]["parallelogram"] = p
        cfg.setdefault("verifier_only_grid_search", {}).update({
            "iteration": int(args.iteration), "family": family, "label": label,
            "center": center, "radius": radius, "seed_config_json": seed["config_json"],
        })
        cfg_path = out_dir / f"{cid}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))
        rows.append({
            "candidate_id": cid,
            "iteration": int(args.iteration),
            "family": family,
            "label": label,
            "checkpoint_path": seed["checkpoint_path"],
            "config_json": str(cfg_path),
            "parallelogram": json.dumps(p),
            "det": det2(p),
            "radius": radius,
        })

    c = [[float(x) for x in row] for row in center]
    v1, v2 = c[0], c[1]

    # 1) scale family
    for k in range(int(args.scale_count)):
        t = 0 if int(args.scale_count) == 1 else (k / (int(args.scale_count) - 1) * 2 - 1)
        s = t * radius
        add_candidate("scale", f"s_{s:+.6g}", [[(1 + s) * x for x in v1], [(1 + s) * x for x in v2]])

    # 2) shear family, approximately area preserving to first order.
    for k in range(int(args.shear_count)):
        t = 0 if int(args.shear_count) == 1 else (k / (int(args.shear_count) - 1) * 2 - 1)
        e = t * radius
        p = [[v1[0] + e * v2[0], v1[1] + e * v2[1]], [v2[0] - e * v1[0], v2[1] - e * v1[1]]]
        add_candidate("shear", f"e_{e:+.6g}", p)

    # 3) coordinate perturbations: axes plus deterministic paired moves.
    for j in range(4):
        for sign in (-1, 1):
            p = copy.deepcopy(c)
            p[j // 2][j % 2] += sign * radius
            add_candidate("coordinate", f"axis{j}_{sign:+d}", p)
    while sum(1 for r in rows if r["family"] == "coordinate") < int(args.coord_count):
        p = copy.deepcopy(c)
        for j in range(4):
            p[j // 2][j % 2] += rng.choice([-1, 1]) * radius * rng.choice([0.25, 0.5, 1.0])
        add_candidate("coordinate", "paired", p)
        if len(rows) > 10000:
            break

    # 4) deterministic random / LHS-like local cloud.
    for k in range(int(args.lhs_count) * 4):
        if sum(1 for r in rows if r["family"] == "lhs") >= int(args.lhs_count):
            break
        p = copy.deepcopy(c)
        for j in range(4):
            p[j // 2][j % 2] += rng.uniform(-radius, radius)
        add_candidate("lhs", f"u{k:03d}", p)

    # Deduplicate by rounded parallelogram.
    seen, deduped = set(), []
    for r in rows:
        key = tuple(round(x, 10) for row in json.loads(r["parallelogram"]) for x in row)
        if key in seen:
            continue
        seen.add(key); deduped.append(r)
    rows = deduped
    write_csv(root / "iterations" / f"iter_{int(args.iteration)}" / "candidates.csv", rows)
    print(json.dumps({"iteration": int(args.iteration), "candidate_count": len(rows), "radius": radius}, indent=2))


def verifier_cmd(row: dict, grid: int, out_dir: Path, solver_limit: int, threads: int, root: Path) -> list[str]:
    return [
        "uv", "run", "python", "scripts/verify_paralellogram_ip.py",
        "--run_id", row["candidate_id"],
        "--eval_gridsize", str(grid),
        "--config_json", row["config_json"],
        "--checkpoint_path", row["checkpoint_path"],
        "--output_dir", str(out_dir),
        "--solver_time_limit", str(solver_limit),
        "--solver_threads", str(threads),
        "--active_edge_chunk_size", "64",
        "--mask_cache_dir", str(root / "mask_cache"),
        "--skip_vertex_cover",
        "--no_plot",
    ]


def run_one(row, args, stage_dir: Path, root: Path) -> dict:
    cid = row["candidate_id"]
    cdir = stage_dir / "outputs" / cid
    log_file = stage_dir / "logs" / f"{cid}.log"
    tmp_dir = stage_dir / "tmp" / cid
    for p in [cdir, log_file.parent, tmp_dir]:
        p.mkdir(parents=True, exist_ok=True)
    if cdir.exists():
        shutil.rmtree(cdir)
        cdir.mkdir(parents=True, exist_ok=True)
    cmd = verifier_cmd(row, int(args.grid), cdir, int(args.solver_limit), int(args.solver_threads), root)
    env = os.environ.copy()
    env.update({
        "TMPDIR": str(tmp_dir), "TEMP": str(tmp_dir), "TMP": str(tmp_dir), "PULP_TMP_DIR": str(tmp_dir),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    t0 = time.time()
    with log_file.open("w") as lf:
        lf.write(json.dumps({"event": "start", "candidate_id": cid, "stage": args.stage, "grid": int(args.grid), "cmd": cmd, "tmpdir": str(tmp_dir), "time": t0}) + "\n")
        try:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, env=env, timeout=int(args.wall_timeout))
            rc = proc.returncode
            err = ""
        except subprocess.TimeoutExpired:
            rc = 124
            err = "wall-timeout"
        lf.write(json.dumps({"event": "end", "candidate_id": cid, "returncode": rc, "elapsed": time.time() - t0, "error": err}) + "\n")
    vc = cdir / "verified_paralellograms_ip.csv"
    rec = dict(row)
    rec.update({
        "stage": args.stage,
        "grid": int(args.grid),
        "pct": pct_from_verification_csv(vc),
        "exit_code": rc,
        "elapsed_sec": round(time.time() - t0, 3),
        "verification_csv": str(vc),
        "log_file": str(log_file),
    })
    return rec


def sweep(args) -> None:
    root = Path(args.root)
    rows = read_csv(Path(args.input_csv))
    stage_dir = root / "iterations" / f"iter_{int(args.iteration)}" / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    status = root / "status.json"
    results = []
    out_csv = stage_dir / f"{args.stage}_results.csv"
    max_workers = max(1, int(args.parallel))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_one, r, args, stage_dir, root): r for r in rows}
        total = len(futs)
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            results.append(rec)
            write_csv(out_csv, results)
            dump_json(status, {"phase": args.stage, "iteration": int(args.iteration), "current": i, "total": total, "latest_candidate": rec.get("candidate_id"), "latest_pct": rec.get("pct"), "ts": time.time()})
            print(json.dumps({"stage": args.stage, "done": i, "total": total, "candidate_id": rec.get("candidate_id"), "pct": rec.get("pct"), "exit": rec.get("exit_code")}), flush=True)
    print("results_csv=" + str(out_csv))


def numeric_pct(row):
    try:
        if row.get("pct") in (None, ""):
            return None
        return float(row["pct"])
    except Exception:
        return None


def select(args) -> None:
    rows = read_csv(Path(args.input_csv))
    valid = [r for r in rows if str(r.get("exit_code")) == "0" and numeric_pct(r) is not None]
    valid.sort(key=lambda r: (float(r["pct"]), r.get("candidate_id", "")))
    selected = []
    if int(args.family_top_k) > 0:
        for fam in sorted({r.get("family", "") for r in valid}):
            selected.extend([r for r in valid if r.get("family", "") == fam][: int(args.family_top_k)])
    selected.extend(valid[: int(args.global_top_k)])
    seen, uniq = set(), []
    for r in selected:
        if r["candidate_id"] in seen:
            continue
        seen.add(r["candidate_id"]); uniq.append(r)
        if len(uniq) >= int(args.max_selected):
            break
    write_csv(Path(args.output_csv), uniq)
    print(json.dumps({"input": len(rows), "valid": len(valid), "selected": len(uniq), "best_pct": float(valid[0]["pct"]) if valid else None}, indent=2))


def refine(args) -> None:
    root = Path(args.root)
    promoted = read_csv(Path(args.input_csv))
    state = load_state(root)
    seed = state["seed"]
    base_cfg = load_json(Path(seed["config_json"]))
    radius = float(args.radius)
    rows = []
    out_dir = root / "iterations" / f"iter_{int(args.iteration)}" / "refined_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas = [(0,0,0,0), (radius,0,0,0), (-radius,0,0,0), (0,radius,0,0), (0,-radius,0,0), (0,0,radius,0), (0,0,-radius,0), (0,0,0,radius), (0,0,0,-radius)]
    for parent in promoted:
        p0 = json.loads(parent["parallelogram"])
        for j, d in enumerate(deltas):
            p = copy.deepcopy(p0)
            for idx, val in enumerate(d):
                p[idx // 2][idx % 2] += val
            cid = f"i{int(args.iteration):03d}_ref_{len(rows):05d}_{parent['candidate_id']}_d{j}"
            cfg = copy.deepcopy(base_cfg)
            cfg["training"]["parallelogram"] = p
            cfg.setdefault("verifier_only_grid_search", {}).update({"iteration": int(args.iteration), "parent_candidate": parent["candidate_id"], "refine_radius": radius})
            cfg_path = out_dir / f"{cid}.json"
            cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))
            rows.append({
                "candidate_id": cid, "iteration": int(args.iteration), "family": parent.get("family", "refine"), "label": "refine",
                "parent_candidate_id": parent["candidate_id"], "checkpoint_path": seed["checkpoint_path"], "config_json": str(cfg_path),
                "parallelogram": json.dumps(p), "det": det2(p), "radius": radius,
            })
    write_csv(root / "iterations" / f"iter_{int(args.iteration)}" / "refined_candidates.csv", rows)
    print(json.dumps({"refined_count": len(rows)}, indent=2))


def report(args) -> None:
    root = Path(args.root)
    state = load_state(root)
    final_rows = read_csv(Path(args.final_csv))
    valid = [r for r in final_rows if str(r.get("exit_code")) == "0" and numeric_pct(r) is not None]
    valid.sort(key=lambda r: (float(r["pct"]), r.get("candidate_id", "")))
    old_best = float(state.get("bestPct", state["baselinePct"]))
    best = valid[0] if valid else None
    improved = bool(best and float(best["pct"]) < old_best)
    beats_baseline = bool(best and float(best["pct"]) < float(state["baselinePct"]))
    if improved:
        state["bestPct"] = float(best["pct"])
        state["bestParallelogram"] = json.loads(best["parallelogram"])
        state["bestSource"] = best["candidate_id"]
    state.setdefault("history", []).append({
        "iteration": int(args.iteration),
        "bestIterationPct": float(best["pct"]) if best else None,
        "improvedState": improved,
        "beatsBaseline": beats_baseline,
        "bestCandidate": best,
    })
    dump_json(root / "state.json", state)
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Iteration {int(args.iteration)} report",
        "",
        f"- baseline pct: `{state['baselinePct']}`",
        f"- previous best pct: `{old_best}`",
        f"- iteration best pct: `{best.get('pct') if best else None}`",
        f"- improved current best: `{improved}`",
        f"- beats baseline: `{beats_baseline}`",
        "- no-improvement policy: `continue; not a workflow failure`",
        "",
    ]
    if best:
        lines += [
            "## Best candidate",
            f"- id: `{best['candidate_id']}`",
            f"- family: `{best.get('family')}`",
            f"- pct: `{best['pct']}`",
            f"- parallelogram: `{best['parallelogram']}`",
            f"- log: `{best.get('log_file')}`",
            "",
        ]
    lines += ["## Top final rows", ""]
    for r in valid[:5]:
        lines.append(f"- `{r['pct']}` `{r.get('family')}` `{r['candidate_id']}`")
    text = "\n".join(lines) + "\n"
    (report_dir / f"iteration_{int(args.iteration)}.md").write_text(text)
    shared = report_dir / "shared_context.md"
    old = shared.read_text() if shared.exists() else "# Shared parallelogram search context\n\n"
    shared.write_text(old + "\n" + text)
    print(text)


def collect_agent_notes(args) -> None:
    root = Path(args.root)
    report_dir = root / "reports"
    shared = report_dir / "shared_context.md"
    text = shared.read_text() if shared.exists() else "# Shared parallelogram search context\n\n"
    text += f"\n# Agent notes after iteration {int(args.iteration)}\n\n"
    patterns = [f"iter_{int(args.iteration):03d}_*.md", f"iter_{int(args.iteration)}_*.md"]
    seen = set()
    for pat in patterns:
        for p in sorted((report_dir / "agents").glob(pat)):
            if p in seen:
                continue
            seen.add(p)
            text += p.read_text()[:4000] + "\n\n"
    shared.write_text(text)
    print("shared_context=" + str(shared))


def init(args) -> None:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for sub in ["logs", "reports", "jobs", "tmp", "iterations", "mask_cache"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in REQUIRED if not p.exists()]
    if missing:
        raise SystemExit("missing required verifier scripts: " + ", ".join(missing))
    if not str(root).startswith("/scratch/htc/npelleriti/pi-sandbox/"):
        print("warning: root is not under /scratch/htc/npelleriti/pi-sandbox", file=sys.stderr)
    state_path = root / "state.json"
    if not state_path.exists() or args.reset_state:
        state = discover_seed(root, float(args.baseline_pct))
    else:
        state = load_state(root)
    dump_json(root / "status.json", {"phase": "initialized", "root": str(root), "bestPct": state.get("bestPct"), "ts": time.time()})
    (root / "reports" / "shared_context.md").write_text(
        "# Shared parallelogram search context\n\n"
        f"- root: `{root}`\n- baselinePct: `{state['baselinePct']}`\n- bestPct: `{state['bestPct']}`\n"
        f"- bestParallelogram: `{json.dumps(state['bestParallelogram'])}`\n\n"
    )
    print(json.dumps({"root": str(root), "bestPct": state.get("bestPct"), "bestParallelogram": state.get("bestParallelogram")}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("--root", required=True); p.add_argument("--baseline-pct", required=True); p.add_argument("--reset-state", action="store_true")
    p = sub.add_parser("make-candidates"); p.add_argument("--root", required=True); p.add_argument("--iteration", required=True); p.add_argument("--base-radius", required=True); p.add_argument("--scale-count", default=25); p.add_argument("--shear-count", default=25); p.add_argument("--coord-count", default=50); p.add_argument("--lhs-count", default=50); p.add_argument("--area-tol", default=0.08); p.add_argument("--norm-tol", default=0.12); p.add_argument("--min-det", default=0.1)
    p = sub.add_parser("sweep"); p.add_argument("--root", required=True); p.add_argument("--iteration", required=True); p.add_argument("--stage", required=True); p.add_argument("--input-csv", required=True); p.add_argument("--grid", required=True); p.add_argument("--parallel", required=True); p.add_argument("--solver-limit", required=True); p.add_argument("--solver-threads", default=1); p.add_argument("--wall-timeout", required=True)
    p = sub.add_parser("select"); p.add_argument("--input-csv", required=True); p.add_argument("--output-csv", required=True); p.add_argument("--family-top-k", default=0); p.add_argument("--global-top-k", default=20); p.add_argument("--max-selected", default=20)
    p = sub.add_parser("refine"); p.add_argument("--root", required=True); p.add_argument("--iteration", required=True); p.add_argument("--input-csv", required=True); p.add_argument("--radius", required=True)
    p = sub.add_parser("report"); p.add_argument("--root", required=True); p.add_argument("--iteration", required=True); p.add_argument("--final-csv", required=True)
    p = sub.add_parser("collect-agent-notes"); p.add_argument("--root", required=True); p.add_argument("--iteration", required=True)
    args = ap.parse_args()
    globals()[args.cmd.replace("-", "_")](args)


if __name__ == "__main__":
    main()
