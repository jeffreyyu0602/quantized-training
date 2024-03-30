import copy
import datetime
import json
import logging
import os
import sys
from pprint import pformat
from typing import List

import wandb
from torch import nn
from torch.nn.utils.parametrize import type_before_parametrizations

__all__ = [
    "get_fused_modules",
    "run_task",
]

logger = logging.getLogger(__name__)

SLURM_LOG_DIR = "slurm_logs"
SLURM_SCRIPT_DIR = "slurm_scripts"
ENV_SETUP_SCRIPT = "setup_shell.sh"

SLURM_ARGS = {
    "job-name": {"type": str, "default": "test"},
    "partition": {"type": str, "default": "gpu"},
    "nodes": {"type": int, "default": 1},
    "time": {"type": str, "default": "48:00:00"},
    "gpus": {"type": str, "default": "1"},
    "cpus": {"type": int, "default": 8},
    "mem": {"type": str, "default": "16GB"},
    "output": {"type": str, "default": None},
    "error": {"type": str, "default": None},
    "exclude": {"type": str, "default": None},
    "nodelist": {"type": str, "default": None},
}

SLURM_NAME_OVERRIDES = {"gpus": "gres", "cpus": "cpus-per-task"}

def write_slurm_script(args, cmd):
    os.makedirs(SLURM_SCRIPT_DIR, exist_ok=True)

    if args.output is None:
        args.output = os.path.join(SLURM_LOG_DIR, args.job_name + ".%j.out")
    if args.error is None:
        args.error = os.path.join(SLURM_LOG_DIR, args.job_name + ".%j.err")
    args.gpus = f"gpu:{args.gpus}" if args.gpus is not None else args.gpus

    with open(os.path.join(SLURM_SCRIPT_DIR, args.job_name + ".sbatch"), "w") as f:
        f.write('#!/bin/bash\n')

        for arg_name in SLURM_ARGS.keys():
            arg_value = vars(args)[arg_name.replace("-", "_")]
            if arg_name in SLURM_NAME_OVERRIDES:
                arg_name = SLURM_NAME_OVERRIDES[arg_name]
            if arg_value is not None:
                f.write(f"#SBATCH --{arg_name}={arg_value}\n")

        f.write('\n')
        f.write('echo "SLURM_JOBID = "$SLURM_JOBID\n')
        f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST\n')
        f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST\n')
        f.write('echo "SLURM_NNODES = "$SLURM_NNODES\n')
        f.write('echo "SLURMTMPDIR = "$SLURMTMPDIR\n')
        f.write('echo "working directory = "$SLURM_SUBMIT_DIR\n')
        f.write('\n')
        f.write('source ' + ENV_SETUP_SCRIPT + '\n')
        f.write('python ' + ' '.join(cmd) + '\n')
        f.write('wait\n')

def write_bash_script(args, cmd):
    filename = "run.sh" if args.run_name is None else args.run_name + ".sh"
    with open(filename, "w") as f:
        f.write('#!/bin/bash\n')
        f.write('python ' + ' '.join(cmd) + '\n')

def run_task(args, run_fn):
    # Set up logging
    if args.log_file == "":
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.log_file = f'logs/{timestamp}.log'
    if args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    logging.basicConfig(
        filename=args.log_file,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=getattr(logging, args.log_level),
    )

    # Create W&B sweep from sweep configuration
    if args.sweep_config:
        with open(args.sweep_config, 'r') as file:
            sweep_configuration = json.load(file)
        args.sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)

    # Write slurm or bash script
    if args.action is not None:
        command = copy.deepcopy(sys.argv)
        if args.sweep_config:
            index = command.index('--sweep_config')
            command[index:index + 2] = ['--sweep_id', args.sweep_id]

        index = command.index(args.action)
        if args.action == "slurm":
            write_slurm_script(args, command[:index])
            return
        elif args.action == "bash":
            write_bash_script(args, command[:index])

    def sweep_function():
        run = wandb.init()
        sweep_args = copy.deepcopy(args)
        for k, v in wandb.config.items():
            if k == "learning_rate" and isinstance(v, int):
                v = float(v) * args.learning_rate
            setattr(sweep_args, k, v)
        logger.info(f"Training/evaluation parameters: {pformat(vars(args))}")
        run_fn(sweep_args)

    # Perform W&B sweep or run single job
    if args.sweep_id is not None:
        wandb.agent(
            args.sweep_id,
            function=sweep_function,
            project=args.project,
            count=args.max_trials,
        )
    else:
        if args.project is not None:
            wandb.init(
                project=args.project,
                name=args.run_name,
                id=args.run_id,
                resume="allow"
            )
        logger.info(f"Training/evaluation parameters: {pformat(vars(args))}")
        run_fn(args)

def get_fused_modules(model: nn.Module, modules_to_fuse: List[nn.Module]):
    module_list = []
    fused_module_list = []
    index = 0

    for name, mod in model.named_modules():
        if type_before_parametrizations(mod) != modules_to_fuse[index]:
            module_list = []
            index = 0

        if type_before_parametrizations(mod) == modules_to_fuse[index]:
            module_list.append(name)
            index += 1
            if index == len(modules_to_fuse):
                fused_module_list.append(module_list)
                module_list = []
                index = 0

    return fused_module_list