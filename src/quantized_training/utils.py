import copy
import datetime
import json
import logging
import os
import sys
from functools import wraps
from pprint import pformat

import wandb

__all__ = [
    "setup_logging",
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
    filename = "train.sh" if args.run_name is None else args.run_name + ".sh"
    with open(filename, "w") as f:
        f.write('#!/bin/bash\n')
        f.write('python ' + ' '.join(cmd) + '\n')

def setup_logging(func):
    @wraps(func)
    def wrapper(args, *func_args, **func_kwargs):
        if args.log_file == "datetime":
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            args.log_file = f'logs/{timestamp}.log'

        if args.log_file is not None:
            dirname = os.path.dirname(args.log_file)
            if dirname != "":
                os.makedirs(dirname, exist_ok=True)

        logging.basicConfig(
            filename=args.log_file,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=getattr(logging, args.log_level),
        )

        # Create W&B sweep from the sweep configuration
        if args.sweep_config is not None:
            if args.sweep_config.endswith(".json"):
                with open(args.sweep_config, 'r') as file:
                    sweep_configuration = json.load(file)
            else:
                from .sweep_config import sweep_configurations
                sweep_configuration = sweep_configurations[args.sweep_config]
            args.sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)

        # Write training commands to a script
        if args.action is not None:
            command = copy.deepcopy(sys.argv)
            if args.sweep_config:
                index = command.index('--sweep_config')
                command[index:index + 2] = ['--sweep_id', args.sweep_id]

            index = command.index(args.action)
            if args.action == "slurm":
                write_slurm_script(args, command[:index])
            elif args.action == "bash":
                write_bash_script(args, command[:index])
            return

        def sweep_fn():
            wandb.init()
            sweep_args = copy.deepcopy(args)
            # W&B does not support specifying the sweep range (min, max) as float types
            # for grid searches. We use int types and multiply the learning rate by the
            # actual value in the sweep function.
            for k, v in wandb.config.items():
                if k == "learning_rate" and isinstance(v, int):
                    v = float(v) * args.learning_rate
                setattr(sweep_args, k, v)
            logger.info(f"Training/evaluation parameters: {pformat(vars(sweep_args))}")
            func(args, *func_args, **func_kwargs)

        if args.sweep_id is not None:
            wandb.agent(args.sweep_id, function=sweep_fn, count=args.max_trials, project=args.project)
        else:
            if args.project is not None:
                wandb.init(
                    project=args.project,
                    name=args.run_name,
                    id=args.run_id,
                    resume="allow"
                )
            logger.info(f"Training/evaluation parameters: {pformat(vars(args))}")
            func(args, *func_args, **func_kwargs)

    return wrapper
