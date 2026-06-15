# ===========================================================================
# Project:      Hadwiger-Nelson
# File:         main.py
# Description:  Starts up a run
# ===========================================================================

import getpass
import os
import shutil
import socket
import sys
import tempfile
from contextlib import contextmanager

import torch
import wandb

from runner import Runner
from utilities import GeneralUtility

debug = "--debug" in sys.argv
defaults = dict(
    # System
    run_id=1,

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
        n_steps=20000,  # total number of parameter updates
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
    ),
)

if not debug:
    # Set everything to None recursively
    defaults = GeneralUtility.fill_dict_with_none(defaults)

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
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
        print('Running on GCP, marking as preemptable.')
        wandb.mark_preempting()  # Note: This potentially overwrites the config when a run is resumed -> problems with tmp_dir

    runner = Runner(config=config, tmp_dir=tmp_dir, debug=debug)
    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
