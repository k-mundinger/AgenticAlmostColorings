# Paper Agent 2 Direct-Fill Construction

This package contains a compact improvement to the paper 5-color almost-coloring
grid. It starts from the paper construction on a `1945 x 1970` parallelogram
grid and applies a verified 91-cell direct-fill patch: each changed cell was
previously the bonus color `5` and is recolored to one of the real colors
`0..4`.

## Result

- Source grid: paper construction `grid.csv`
- Source grid SHA256: `d85a38934ce3d54870fd322662087e0e6fd58c9e38d8b8e529967a7c6f5788df`
- Grid shape: `1945 x 1970`
- Changed cells: `91`
- Bonus count: `143135 -> 143044`
- Bonus-color fraction: `3.735596936% -> 3.733221980%`
- Independent discrete verifier result: success
- Real-color directed unit-distance conflicts after patch: `0`

## Files

- `parallelogram.csv`: paper parallelogram vectors and subdivisions.
- `verification/fixed_coloring_1945x1970_agent2_direct_fill.npz`: compressed
  patched coloring grid.
- `verification/agent2_direct_fill_diff.csv`: exact 91-cell patch.
- `verification/agent2_direct_fill_diff.json`: JSON version of the patch.
- `verification/agent2_direct_fill_independent_verification.json`: independent
  verification output for the patch.
- `agent_run/instruction.md`: instruction given to Agent 2.
- `agent_run/report.md`: Agent 2's report explaining the search.
- `agent_run/artifacts.json`: Agent 2's artifact manifest.
- `agent_run/analysis_summary.json`: Agent 2's local-search summary.

Load the patched coloring with:

```python
import numpy as np

grid = np.load(
    "verification/fixed_coloring_1945x1970_agent2_direct_fill.npz"
)["fixed_coloring"]
```

## How It Was Found

This candidate was found by Agent 2 in round 1 of the
`papers-concept-exploration-loop` workflow. The agent was instructed to work
from the paper `grid.csv` and `parallelogram.csv`, avoid neural checkpoint
artifacts, and report both improvements and negative evidence.

Agent 2 built a parser for the full paper grid, rebuilt a rectangular torus
unit-distance mask, and used deterministic local search with seed `1729`.
It found 190 bonus cells where at least one real color was absent from the
unit-distance neighborhood. A greedy independent-set pass accepted 91 compatible
direct fills, producing the CSV diff preserved in `verification/`.

The agent also reported negative evidence: naive 5x subsampling was not a good
proxy, most bonus cells were tightly constrained, and simple boundary-copy moves
usually introduced immediate conflicts. See `agent_run/report.md` for details.

## Verification

The patch was independently checked by
`scripts/verify_agent2_direct_fill_candidate.py`, which does not import the
agent's generated analysis script. It validates the diff against the original
paper grid, rebuilds the unit-distance mask from the parallelogram, checks all
changed-cell neighborhoods, and recomputes full-grid same-real-color directed
conflicts with FFTs.

The Slurm verification job was `1333310`; it completed with exit code `0`.
The committed JSON result is
`verification/agent2_direct_fill_independent_verification.json`.

Re-run the verifier with the original paper `grid.csv` available:

```bash
python3 scripts/verify_agent2_direct_fill_candidate.py \
  --grid-csv constructions/paper_almost_5_coloring_2d/grid.csv \
  --parallelogram-csv constructions/paper_agent2_direct_fill_1945x1970/parallelogram.csv \
  --diff-csv constructions/paper_agent2_direct_fill_1945x1970/verification/agent2_direct_fill_diff.csv \
  --output-dir /tmp/agent2_direct_fill_verification
```

This is a discrete paper-grid/cell-mask verification. It is not a separate
continuous proof of the construction.

## Checksums

```text
6994b3d7198fde32dc12807a3a2241720171b97c3675fc5144cce2e0caa14ac8  verification/fixed_coloring_1945x1970_agent2_direct_fill.npz
4a0d5b0b26264f68fd7f4f2d2b1e243ce95ec50799a4add2577c5a19c2353ed7  verification/agent2_direct_fill_diff.csv
17b8c01c0951ba40c38329f796c54deea4ba276b22bf2ff51413d056da76fde6  verification/agent2_direct_fill_independent_verification.json
c553eaab020538bafb471832f7de52050224c10e3448bae6cb35929da3524a5d  parallelogram.csv
```
