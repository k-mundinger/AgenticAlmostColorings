# Frontier Post-142342 Construction

This package contains the best one-copy paper-grid candidate found by the
`incumbent-frontier-ilp-loop` workflow. It starts from the paper construction on
the `1945 x 1970` parallelogram grid and applies a paper-relative diff produced
in round 20 by Agent 1.

## Result

- Candidate name: `combined_post142344_t10_rank49_t12_rank65`
- Source workflow job: `incumbent-frontier-ilp-loop`, Slurm job `1339449`
- Source agent: round 20, Agent 1
- Source grid: paper construction `grid.csv`
- Source grid SHA256: `d85a38934ce3d54870fd322662087e0e6fd58c9e38d8b8e529967a7c6f5788df`
- Grid shape: `1945 x 1970`
- Paper-relative changed cells: `7371`
- Incremental changed cells versus post142344 base: `31`
- Bonus count: `143135 -> 142342`
- Bonus-color fraction: `3.735596936% -> 3.714900891%`
- Agent-side independent audit: passed
- Real-color directed unit-distance conflicts after patch: `0`

## Files

- `parallelogram.csv`: paper parallelogram vectors and subdivisions.
- `verification/fixed_coloring_1945x1970_frontier_post142342.npz`:
  compressed patched coloring grid.
- `verification/frontier_post142342_candidate_vs_paper.csv`: full
  paper-relative diff.
- `verification/frontier_post142342_incremental_vs_post142344.csv`:
  31-row incremental diff versus the round 19 `post142344` base.
- `verification/frontier_post142342_independent_audit.json`: audit output for
  the candidate package.
- `verification/frontier_post142342_count_table.csv`: final color counts.
- `agent_run/report.md`: Agent 1's round 20 report explaining the search.

Load the patched coloring with:

```python
import numpy as np

grid = np.load(
    "verification/fixed_coloring_1945x1970_frontier_post142342.npz"
)["fixed_coloring"]
```

## How It Was Found

The candidate was produced by the `incumbent-frontier-ilp-loop` after 20 rounds
of one-copy local ILP search. Agent 1 first confirmed the round 19
`post142344` one-copy candidate, then regenerated threshold-10/11/12 local
components from that base.

The useful improvements were two disjoint, verified local increments:

- `t10_rank49`: 9 incremental rows
- `t12_rank65`: 22 incremental rows

Combining those increments gave `combined_post142344_t10_rank49_t12_rank65`,
reducing color 5 from `142344` to `142342`. The agent report in
`agent_run/report.md` preserves the solve parameters, selected cases, negative
evidence, and follow-up recommendations.

## Verification

The committed audit JSON reports:

- `audit_pass: true`
- `candidate_sha256:
  71a86e29e95f739cedce70b696b8646f48f376294adcac10786f821b6ff3d804`
- `incremental_sha256:
  183d8b33a44aa91eb559517a9dbc1c34df19b21c694dc2d550ea90a6f8f77592`
- `color5_count: 142342`
- `directed_conflicts_total: 0`
- full/incremental array agreement
- no old-value mismatches, duplicate rows, or source-coordinate overlaps

This is a discrete paper-grid verification under the workflow's rebuilt
9280-offset torus mask. It is not a separate continuous proof of the
construction. Before treating this as the canonical repository best, run a fresh
sibling audit from the committed diff.

## Checksums

```text
d87414aae0a98b028a8bd528ed84d29f3157955b4e8765ccd294fd886f209d81  verification/fixed_coloring_1945x1970_frontier_post142342.npz
71a86e29e95f739cedce70b696b8646f48f376294adcac10786f821b6ff3d804  verification/frontier_post142342_candidate_vs_paper.csv
b707c009c43e03ff7d9edb09c7be614e6b5fd311ffc3ab726f91ac338bd34cd3  verification/frontier_post142342_count_table.csv
183d8b33a44aa91eb559517a9dbc1c34df19b21c694dc2d550ea90a6f8f77592  verification/frontier_post142342_incremental_vs_post142344.csv
5f1100449f7c3427846ff007b36b70762653f855cd23ed36d9b8aceda2975a98  verification/frontier_post142342_independent_audit.json
c553eaab020538bafb471832f7de52050224c10e3448bae6cb35929da3524a5d  parallelogram.csv
a18d142f6bed953da252fbc38fd19839f2ceaf6ca71d868cf159c384727b6665  agent_run/report.md
```
