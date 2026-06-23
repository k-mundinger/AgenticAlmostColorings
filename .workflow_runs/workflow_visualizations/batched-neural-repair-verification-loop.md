# batched-neural-repair-verification-loop

```mermaid
flowchart TD
  start([workflow_run]) --> preflight["preflight<br/>shell<br/>source worktree, shared venv,<br/>ensemble_eval.py, smoke trainer"]
  preflight --> capacity["capacity_calibration<br/>shell<br/>timed one-GPU ensemble capacity sweep"]
  capacity --> seed["seed_initial_instructions<br/>agent, xhigh<br/>create four starting briefs"]
  seed --> validateSeed["validate_seed_instructions<br/>shell<br/>schema check and install round-1 prompts"]
  validateSeed --> loop{"exploration_loop<br/>loop: 1-20 rounds"}

  loop --> roundPreflight["round_preflight<br/>shell<br/>create per-agent worktrees,<br/>prompts, GPU lock, scratch dirs"]
  roundPreflight --> fanout{{"parallel_explorers<br/>four async_shell jobs"}}

  fanout --> agent1["agent_1_async<br/>geometry/parallelogram<br/>thinking: high<br/>timeout: 1h<br/>allowFailure: true"]
  fanout --> agent2["agent_2_async<br/>loss/schedule<br/>thinking: high<br/>timeout: 1h<br/>allowFailure: true"]
  fanout --> agent3["agent_3_async<br/>architecture/capacity<br/>thinking: medium<br/>timeout: 1h<br/>allowFailure: true"]
  fanout --> agent4["agent_4_async<br/>repair/verification<br/>thinking: xhigh<br/>timeout: 1h<br/>allowFailure: true"]

  agent1 --> collect["collect_round_outputs<br/>shell<br/>read reports, artifacts, async DB;<br/>treat failures/timeouts as evidence"]
  agent2 --> collect
  agent3 --> collect
  agent4 --> collect

  collect --> reflect["reflect_and_spawn<br/>agent, xhigh<br/>synthesize four outputs and<br/>emit next four instructions"]
  reflect --> validateReflection["validate_reflection_and_update_state<br/>shell<br/>schema check and update current prompts"]
  validateReflection -. "next round until 20" .-> roundPreflight
  validateReflection --> finalSynth["final_synthesis<br/>agent, xhigh<br/>strongest candidates, failed ideas,<br/>fanout plan"]
  finalSynth --> validateFinal["validate_final_synthesis<br/>shell<br/>check final_synthesis.md and<br/>final_fanout_plan.json"]
  validateFinal --> done([done])

  classDef shell fill:#eef5ff,stroke:#4d78b8,color:#111;
  classDef agent fill:#fff3d6,stroke:#b8871f,color:#111;
  classDef async fill:#f1e9ff,stroke:#7a4cc2,color:#111;
  classDef control fill:#eaf7ea,stroke:#3f8d46,color:#111;
  class preflight,capacity,validateSeed,roundPreflight,collect,validateReflection,validateFinal shell;
  class seed,reflect,finalSynth agent;
  class agent1,agent2,agent3,agent4 async;
  class loop,fanout control;
```

## Operational Flow

1. Preflight creates a detached source worktree from `origin/konrad-manual-sweep`, builds one shared venv, installs the scratch-only `ensemble_eval.py` helper, and runs a tiny smoke training check.
2. Capacity calibration runs a bounded CUDA ensemble sweep and writes `state/capacity_calibration.json`.
3. A seed agent turns the design brief and calibration into four starting search briefs.
4. Each loop round creates four isolated worktrees and launches four async explorer jobs in parallel.
5. Explorer failures and timeouts are intentionally collected as search evidence rather than hard-stopping the workflow.
6. A reflection agent synthesizes the round and writes the next four instructions.
7. After the loop budget, a final synthesis agent writes `final_synthesis.md` and `final_fanout_plan.json`, then a shell gate validates those outputs.

## Key Runtime Paths

- Workflow spec: `.pi/workflows/batched-neural-repair-verification-loop.json`
- Helper script: `.pi/scripts/batched_neural_workflow.py`
- Slurm wrapper: `slurm/run_batched_neural_repair_verification_loop.sbatch`
- Current output root: `/scratch/htc/npelleriti/pi-sandbox/batched-neural-repair-verification-loop/job_1339429`
- Async DB: `{outputRoot}/jobs/batched-neural-repair-verification-loop.sqlite`
