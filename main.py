# ===========================================================================
# Project:      Hadwiger-Nelson
# File:         main.py
# Description:  Starts up a run
# ===========================================================================

import getpass
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from contextlib import contextmanager

import torch
import wandb
from wandb.errors import Error as WandbError

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

from runner import Runner
from utilities import GeneralUtility

debug = "--debug" in sys.argv
fast_train = "--fast-train" in sys.argv
if fast_train:
    os.environ["WANDB_MODE"] = "disabled"


def _cli_value(flag: str, cast):
    if flag not in sys.argv:
        return None
    idx = sys.argv.index(flag)
    if idx + 1 >= len(sys.argv):
        raise ValueError(f"Missing value for {flag}")
    return cast(sys.argv[idx + 1])


def _parse_parallelogram(raw: str):
    rows = []
    for row in raw.split(";"):
        rows.append([float(item.strip()) for item in row.split(",") if item.strip()])
    if len(rows) != 2 or any(len(row) != 2 for row in rows):
        raise ValueError(f"Expected --parallelogram 'x1,y1;x2,y2', got {raw!r}")
    return rows


class ConfigDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def update(self, *args, **kwargs):
        updates = dict(*args, **kwargs)
        for key, value in updates.items():
            self[key] = _to_config(value)


def _to_config(value):
    if isinstance(value, dict):
        return ConfigDict({key: _to_config(val) for key, val in value.items()})
    return value


def _to_plain(value):
    if isinstance(value, dict):
        return {key: _to_plain(val) for key, val in value.items()}
    if hasattr(value, "items"):
        return {key: _to_plain(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(val) for val in value]
    if isinstance(value, list):
        return [_to_plain(val) for val in value]
    return value


defaults = dict(
    # System
    run_id=1,
    seed=None,

    # Problem definition
    problem_name='HadwigerNelson',
    dim=2,
    n_colours=6,

    # Optimizer definition
    optimizer=dict(
        name='AdamW',
        learning_rate=0.001,    # Linear Schedule from learning_rate to 0 after 5% warmup iterations
        weight_decay=0.1
    ),

    # Training definition
    training=dict(
        # General
        n_steps=10000,  # total number of parameter updates
        batch_size=2048,  # Batch size for training
        grid_input_scale=1,  # Scale of the grid as how it is input to the network
        loss_fn="log_prob",
        grid_sizes=(6,6),  # Must be a tuple with the grid sizes for each dimension (var dim)
        p_norm=2,  # The norm that induces the distance w.r.t which we sample "unit distance" points
        n_circle_points=8, # number of proximity points to sample for each colour
        temperature=0.0,  # circle-point aggregation: <0 hard max, 0 plain mean, >0 softmax-weighted
        good_coloring=True,  # for lagrangian term for last colour
        good_coloring_weight=0.01,
        parallelogram=[[2.0, 1.0], 
                       [1.0, 2.0]],  # None or a list of vectors (they are the rows!)
        trainable_parallelogram=1
    ),

    # Model definition
    model=dict(
        name='ResMLP',
        n_hidden_layers=2,
        n_hidden_units=64, 
        activation='sin',
        initialization="siren",  # If None, uses default value.
        disable_residual_connections=True,  # If True, disables residual connections between hidden layers,
    ),

    metrics=dict(
        plot_grid_size=512, # grid size for the plots
        val_grid_size=128, # grid size for the metrics
        n_circle_points=128, # the same for plots and metrics
        log_metrics_every_k_steps=1000,  # how often to log metrics
        log_imgs_every_k_steps=1000,  # how often to log images
        log_model_every_k_steps=100000,  # how often to log the model
        enable_wandb_logging=True,
        enable_eval=True,
        enable_plots=True,
        save_intermediate_models=True,
        sync_models_to_wandb=True,
        save_final_model=True,
        loss_log_every_k_steps=1000,
    ),
)

if fast_train:
    defaults["metrics"].update(
        enable_wandb_logging=False,
        enable_eval=False,
        enable_plots=False,
        save_intermediate_models=False,
        sync_models_to_wandb=False,
        save_final_model=True,
        loss_log_every_k_steps=1000,
    )

for cli_flag, config_key in (
    ("--n-steps", "n_steps"),
    ("--batch-size", "batch_size"),
    ("--n-circle-points", "n_circle_points"),
):
    override = _cli_value(cli_flag, int)
    if override is not None:
        defaults["training"][config_key] = override

for cli_flag, config_key in (
    ("--learning-rate", "learning_rate"),
    ("--weight-decay", "weight_decay"),
):
    override = _cli_value(cli_flag, float)
    if override is not None:
        defaults["optimizer"][config_key] = override

for cli_flag, config_key in (
    ("--loss-fn", "loss_fn"),
):
    override = _cli_value(cli_flag, str)
    if override is not None:
        defaults["training"][config_key] = override

for cli_flag, config_key in (
    ("--temperature", "temperature"),
    ("--good-coloring-weight", "good_coloring_weight"),
):
    override = _cli_value(cli_flag, float)
    if override is not None:
        defaults["training"][config_key] = override

for cli_flag, config_key in (
    ("--n-hidden-layers", "n_hidden_layers"),
    ("--n-hidden-units", "n_hidden_units"),
):
    override = _cli_value(cli_flag, int)
    if override is not None:
        defaults["model"][config_key] = override

loss_log_every = _cli_value("--loss-log-every", int)
if loss_log_every is not None:
    defaults["metrics"]["loss_log_every_k_steps"] = loss_log_every

parallelogram_override = _cli_value("--parallelogram", str)
if parallelogram_override is not None:
    defaults["training"]["parallelogram"] = _parse_parallelogram(parallelogram_override)

trainable_parallelogram_override = _cli_value("--trainable-parallelogram-step", int)
if trainable_parallelogram_override is not None:
    defaults["training"]["trainable_parallelogram"] = trainable_parallelogram_override

if "--freeze-parallelogram" in sys.argv:
    defaults["training"]["trainable_parallelogram"] = 0

seed_override = _cli_value("--seed", int)
if seed_override is not None:
    defaults["seed"] = seed_override

output_root = _cli_value("--output-root", str) or "models"
local_run_id = _cli_value("--local-run-id", str)

if not debug:
    # Set everything to None recursively
    defaults = GeneralUtility.fill_dict_with_none(defaults)

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

# Configure wandb logging
if fast_train:
    config = _to_config(defaults)
else:
    wandb_project = os.getenv('WANDB_PROJECT', 'test-000')
    wandb_entity = os.getenv('WANDB_ENTITY', None)
    wandb.init(
        config=defaults,
        project=wandb_project,  # automatically changed in sweep
        entity=wandb_entity,    # automatically changed in sweep
    )
    config = wandb.config
config = GeneralUtility.update_config_with_default(config, defaults)

# Check if config contains any parameters that are not in defaults, then this should raise an exception
has_unknown_params, params = GeneralUtility.config_has_unknown_params(config, defaults)
assert not has_unknown_params, f"Unknown parameters {params} in config."

ngpus = torch.cuda.device_count()
if ngpus > 0:
    config.update(dict(device='cuda:0'))
else:
    config.update(dict(device='cpu'))


@contextmanager
def tempdir():
    username = getpass.getuser()
    tmp_root = '/scratch/local/' + username
    tmp_path = os.path.join(tmp_root, 'tmp')
    if os.path.isdir('/scratch/local/') and not os.path.isdir(tmp_root):
        os.makedirs(tmp_root, exist_ok=True)
    if os.path.isdir(tmp_root):
        if not os.path.isdir(tmp_path): os.makedirs(tmp_path, exist_ok=True)
        path = tempfile.mkdtemp(dir=tmp_path)
    else:
        assert 'htc-' not in os.uname().nodename, "Not allowed to write to /tmp on htc- machines."
        path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write('Failed to clean up temp dir {}'.format(path))


with tempdir() as tmp_dir:
    # Check if we are running on the GCP cluster, if so, mark as potentially preempted
    is_htc = 'htc-' in os.uname().nodename
    is_gcp = 'gpu' in os.uname().nodename and not is_htc
    if is_gcp:
        # In disabled/offline modes, wandb.run may be unavailable.
        # Guard this call so local pipelines can run without W&B.
        if wandb.run is not None:
            print('Running on GCP, marking as preemptable.')
            try:
                wandb.mark_preempting()  # Potentially overwrites config on resume.
            except WandbError as e:
                print(f'Skipping wandb.mark_preempting(): {e}')

    runner = Runner(config=config, tmp_dir=tmp_dir, debug=debug)
    runner.run()

    run_config = _to_plain(config)
    if run_config.get("training", {}).get("parallelogram") is not None and hasattr(runner.problem.model, "inv_transf_matrix"):
        parallelogram = torch.linalg.inv(runner.problem.model.inv_transf_matrix.detach()).cpu().tolist()
        run_config["training"]["parallelogram"] = parallelogram
    with open(os.path.join(tmp_dir, "pipeline_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    # Save a persistent copy of all outputs to models/run_id/ before the tempdir is deleted.
    # Fast/local modes may have no real W&B run, so synthesize a stable local id.
    run_id = wandb.run.id if wandb.run is not None else (local_run_id or f"local_{int(time.time())}_{os.getpid()}")
    local_dir = os.path.join(output_root, run_id)
    os.makedirs(local_dir, exist_ok=True)
    for item in os.listdir(tmp_dir):
        s = os.path.join(tmp_dir, item)
        d = os.path.join(local_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    sys.stdout.write(f"Saved persistent local copy of all outputs to {local_dir}.\n")

    # Close wandb run
    wandb_dir_path = wandb.run.dir if wandb.run is not None else None
    if wandb.run is not None:
        wandb.join()

    # Delete local W&B files if possible. In disabled/offline setups this path
    # can be a protected shared tmp location; skip cleanup gracefully.
    if wandb_dir_path and os.path.exists(wandb_dir_path):
        try:
            shutil.rmtree(wandb_dir_path)
        except (OSError, PermissionError) as e:
            sys.stderr.write(f"Skipping wandb dir cleanup ({wandb_dir_path}): {e}\n")
