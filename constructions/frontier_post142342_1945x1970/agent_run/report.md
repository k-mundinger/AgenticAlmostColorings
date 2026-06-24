# Round 20 agent_1 report

Status: completed

## Result

Confirmed the Round 19 one-copy candidate, then continued from it as the `post142344_combined_r19` base. Best new candidate is:

- Name: `combined_post142344_t10_rank49_t12_rank65`
- Color-5 count: `142342`
- Beats Round 19 base `142344` by `2` cells; beats campaign plus744 `142391` by `49` cells.
- Full diff SHA256: `71a86e29e95f739cedce70b696b8646f48f376294adcac10786f821b6ff3d804`
- Incremental-vs-post142344 SHA256: `183d8b33a44aa91eb559517a9dbc1c34df19b21c694dc2d550ea90a6f8f77592`
- Independent audit: passed; zero directed conflicts; full/incremental arrays agree; old-value/duplicate/overlap checks passed.

Candidate package:

- Full diff: `/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_20/agent_1/experiments/post142344_onecopy/combined_candidates/t10_rank49_plus_t12_rank65/combined_post142344_t10_rank49_t12_rank65_candidate_vs_paper.csv`
- Incremental diff: `/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_20/agent_1/experiments/post142344_onecopy/combined_candidates/t10_rank49_plus_t12_rank65/combined_post142344_t10_rank49_t12_rank65_incremental_vs_post142344_combined_r19.csv`
- Audit JSON: `/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_20/agent_1/experiments/post142344_onecopy/combined_candidates/t10_rank49_plus_t12_rank65/combined_post142344_t10_rank49_t12_rank65_independent_audit.json`
- Count table: `/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_20/agent_1/experiments/post142344_onecopy/combined_candidates/t10_rank49_plus_t12_rank65/combined_post142344_t10_rank49_t12_rank65_count_table.csv`

## Startup audit of Round 19 candidate

Audit path: `/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_20/agent_1/experiments/post142344_onecopy/audit/round19_post142344_candidate_independent_audit.json`

Expected checks all passed:

- Color5: `142344`
- Full SHA: `8e424885d59ef09f776e97c58c2cb3892ed919af0b55a33e24b801b2f69e9a70`
- Incremental SHA: `3c3fb6c6eb18e5da676798a3a6adf29fa4c4a8b99ff7ff7f5ebcd988b9f99d64`
- Directed conflicts total: `0`
- Full/incremental array mismatch count: `0`
- Old-value, duplicate, source-overlap checks: pass
- Audit wall: `24 s`; max RSS: `58896 KB`

## Commands and parameters

All commands used:

```bash
export TMPDIR=/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
export PULP_TMP_DIR=$TMPDIR
```

Startup audit, wall `24 s`:

```bash
/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/shared_venv/bin/python \
  /scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_20/agent_1/experiments/post142344_onecopy/audit/audit_round19_candidate_post142344.py
```

Post142344 component regeneration, wall `13 s`:

```bash
/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/shared_venv/bin/python \
  /scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_15/agent_1/experiments/post142354_onecopy/incumbent_frontier_decomposition.py analyze-base \
  --grid-csv /workspace/constructions/paper_almost_5_coloring_2d/grid.csv \
  --parallelogram-csv /workspace/constructions/paper_almost_5_coloring_2d/parallelogram.csv \
  --incumbent-diff /scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_19/agent_1/experiments/post142347_onecopy/combined_candidates/t10_rank66_plus_t10_rank78/combined_post142347_t10_rank66_t10_rank78_candidate_vs_paper.csv \
  --incumbent-label post142344_combined_r19 \
  --out-dir /scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_20/agent_1/experiments/post142344_onecopy/base_analysis \
  --thresholds 10 11 12 --ilp-max-cells 1500 \
  --tmp-dir /scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/tmp \
  --save-top-components 200 --verify-paper
```

Regenerated component max sizes: t10 `778`, t11 `815`, t12 `870`; no component exceeded `ilpMaxCells=1500`.

Case selection, selected 75 cases:

```bash
/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/shared_venv/bin/python \
  /scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_20/agent_1/experiments/post142344_onecopy/case_selection/select_cases_r20.py
```

Solve driver, wall `918 s`:

```bash
/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/shared_venv/bin/python \
  /scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_20/agent_1/experiments/post142344_onecopy/solves_focused/run_selected_solves_r20.py
```

Each individual solve used `solve-base` with CBC threads `1`, time limit `900`, `--no-split`, `--objective-cut-gain 1`, `--ilp-max-cells 1500`, dense guards `--max-edge-constraints 1000000 --max-component-edge-constraints 1000000 --max-estimated-mps-mb 2048`, and `--log-max-rss`. Per-case exact commands are in `post142344_t*_rank*_nosplit_cut1_r20.command.txt` under the solve output directory.

Combination/audit of verified disjoint increments, wall `25 s`:

```bash
/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/shared_venv/bin/python \
  /scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_20/agent_1/experiments/post142344_onecopy/combined_candidates/combine_and_audit_r20.py
```

## Solve aggregate and negative evidence

Solve aggregate: `/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_20/agent_1/experiments/post142344_onecopy/solves_focused/post142344_selected75_aggregate_summary.json`

- Cases run: `75`
- Candidate verified: `4`
- Infeasible objective cut: `71`
- Exact registry skips: `0`
- Dense/MPS guard skips: `0`
- Directed conflicts: all `0`
- Max RSS: `326204 KB` during solves; combined audit max RSS `660016 KB`
- Max estimated MPS: `2.616 MiB`; max color inequality count: `2524`

The four verified single-case candidates all reached color5 `142343`. Three were the same 9-row increment (`t10_rank49`, `t11_rank42`, `t12_rank8`, SHA `494fa4f5d72847143066428d86bd9b730e97e3c35aed88212d782fb7d94abda2`). The other was `t12_rank65`, a 22-row increment with SHA `52609d0f3efd78792268a0642cd98fd8bd7f94c26b64926a05577d91d29803ad`. These two unique increments were disjoint and combined cleanly to color5 `142342`.

No selected case was skipped as too large. In the regenerated post142344 component table, all 250 enumerated t10/t11/t12 components were below the `1500` cell cap; remaining unsearched cases are due to the 75-case final-round selection budget, not memory/size guards.

## Pre-skip / registry behavior

Pre-skip decisions: `/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_20/agent_1/experiments/post142344_onecopy/solves_focused/post142344_selected75_pre_skip_decisions.csv`

Registry policy was exact-skip only. There were no exact skips because the base is the new post142344 diff. Older post142347/post142350 signatures were treated as warning/run evidence, not as skip authorization.

## Next instruction

Promote only after another independent audit of `combined_post142344_t10_rank49_t12_rank65` confirms SHA `71a86e29e95f739cedce70b696b8646f48f376294adcac10786f821b6ff3d804`, incremental SHA `183d8b33a44aa91eb559517a9dbc1c34df19b21c694dc2d550ea90a6f8f77592`, color5 `142342`, zero conflicts, and old-value/overlap checks. Then use it as the post142342 base: recompute t10/t11/t12 components, target final color5 `<=142341`, seed around the new atoms (`t10_rank49`/`t11_rank42`/`t12_rank8` and `t12_rank65`), and refresh periodic accounting to repeated-base color5 `284684` with promotion gate `<=284683`.

## Artifacts

Machine-readable artifact manifest: `/scratch/htc/npelleriti/pi-sandbox/incumbent-frontier-ilp-loop/job_1339449/round_20/agent_1/artifacts.json`
