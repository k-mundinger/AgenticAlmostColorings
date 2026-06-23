# Status: completed

## Instruction addressed
Computational/local-search extension grounded only in `grid.csv` and `parallelogram.csv`. I built a parser for the 1945-by-1970 paper grid, computed a rectangular torus unit-distance mask, and ran short deterministic local searches with seed `1729`. No neural checkpoints or verifier/oracle files were used or modified.

## Paper-grounded observations
- `grid.csv` row order is `i` along `v1` outer, `j` along `v2` inner; coordinates match `i*v1/1945 + j*v2/1970` with max CSV rounding error `8.81e-7`.
- Paper color counts: `{0:787790, 1:656953, 2:788588, 3:832312, 4:622872, 5:143135}`; bonus fraction `0.03735596936`.
- The full-resolution conflict mask has 9280 offsets. FFT verification found zero same-real-color unit-distance hits in the original grid.

## Concrete ideas or candidate construction changes
- Candidate improvement: greedily fill unit-distance-safe bonus cells.
  - 190 bonus cells have at least one real color absent from their unit-distance neighborhood.
  - A deterministic greedy independent-set pass accepts 91 such fills without creating conflicts among newly filled cells.
  - Verified result: bonus count `143135 -> 143044`, bonus fraction `0.03735596936 -> 0.03733221980`, with zero directed same-color hits after recomputation.
  - Diff artifacts: `experiments/paper_local_search/out/full_1945x1970_direct_fill_candidate_diff.{json,csv}`.
- Additional small move: one one-step repair changes `(579,580): 5->0` and `(1,844): 0->1`; independently recomputed zero conflicts and bonus count `143134`. This is separate from the 91-fill candidate and should be tested for compatibility before combining.

## Tests/experiments run or proposed, with exact commands and paths
From worktree `/scratch/htc/npelleriti/pi-sandbox/papers-concept-exploration-loop/job_1319209/worktrees/round_1/agent_2`:

```bash
UV_LINK_MODE=copy uv run python experiments/paper_local_search/analyze_paper_grid.py \
  --coarse-factors 5 --device cpu \
  --out-dir experiments/paper_local_search/out \
  --repair-limit 500 --max-blockers 8
```

Key paths:
- Script: `experiments/paper_local_search/analyze_paper_grid.py`
- Combined summary: `experiments/paper_local_search/out/analysis_summary.json`
- Full-resolution summary: `experiments/paper_local_search/out/full_1945x1970_summary.json`
- Candidate direct-fill CSV diff: `experiments/paper_local_search/out/full_1945x1970_direct_fill_candidate_diff.csv`
- One-step verification: `experiments/paper_local_search/out/full_1945x1970_one_step_candidate_verification.json`

## Evidence against weak ideas or no-improvement findings
- Naive 5x coarsening by subsampling (`389x394`) is not a useful verifier proxy: it has 15,295 undirected same-color conflict edges and no singly recolorable bonus cells.
- Most bonus cells are tightly constrained: full-resolution median minimum blocker count over colors is 340; 135,790 of 143,135 bonus cells have minimum blocker count at least 20.
- Boundary-copy moves often create immediate conflicts; sampled examples created 5, 23, 27, and 41 conflicts when a bonus cell was recolored to an adjacent real color.

## Risks and assumptions
- The mask implementation follows the repository verifier style but is a new rectangular adaptation; the candidate should be checked by an independent verifier before treating it as formal.
- The 91-fill greedy result is not guaranteed maximum. Exact MIS/ILP on the 190 safe cells may accept more.
- The one-step repair was verified alone, not combined with the 91-fill candidate.

## Artifact paths worth preserving
- `experiments/paper_local_search/analyze_paper_grid.py`
- `experiments/paper_local_search/out/analysis_summary.json`
- `experiments/paper_local_search/out/full_1945x1970_direct_fill_candidate_diff.csv`
- `experiments/paper_local_search/out/full_1945x1970_direct_fill_candidate_diff.json`
- `experiments/paper_local_search/out/full_1945x1970_one_step_candidate_verification.json`
- `experiments/paper_local_search/out/mask_cache/`

## Recommended follow-up instruction for a future agent
Independently verify the 91-cell CSV diff with a clean rectangular-grid verifier, then solve an exact maximum independent set / small ILP over the 190 singly safe bonus fills plus compatible one-step repairs. If verified, emit a full patched `grid.csv` or compact patch format and benchmark whether the bonus fraction remains conflict-free under the official verification convention.
