import argparse
import copy
import datetime
import importlib
import logging
import os
import re
import sys
import wandb
from pprint import pformat


logger = logging.getLogger(__name__)

SLURM_LOG_DIR = "slurm_logs"
SLURM_SCRIPT_DIR = "slurm_scripts"
BASH_SCRIPT_DIR = "bash_scripts"
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

def parse_args():
    parser = argparse.ArgumentParser(description="Task runner", allow_abbrev=False)
    parser.add_argument("--module", required=True, type=str, help="Task module to run.")
    parser.add_argument(
        '--project_name',
        type=str,
        default=None,
        help=(
            'Optionally provide the name of the project for the project parameter '
            '(project) where you want the output of the W&B Run to be stored.'
        )
    )
    parser.add_argument(
        '--sweep_config',
        type=str,
        default=None,
        help='Path to JSON file that stores sweep configuration.'
    )
    parser.add_argument(
        '--sweep_id',
        type=str,
        default=None,
        help='W&B sweep ID that includes the the entity name and the project name.'
    )
    parser.add_argument(
        '--run_id',
        type=str,
        default=None,
        help='A unique ID for a wandb run, used for resuming.'
    )
    parser.add_argument("--job_count", type=int, default=None, help="Maximum number of runs to try for each batch job.")
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Set the logging file. If not specified, the log will be printed to default location."
    )
    parser.add_argument(
        "--write_script",
        choices=['bash', 'slurm'],
        default=None,
        help="Write a script for the given configuration. Choose either 'bash' or 'slurm' script.",
    )
    for k, v in SLURM_ARGS.items():
        parser.add_argument("--" + k, **v)
    args, training_args = parser.parse_known_args()
    return args, training_args

def get_time_in_hour(time_str):
    if (match := re.match(r"(\d+)-(\d+)(?:(\d+):)?(?:(\d+):)?", time_str)):
        days = int(match.group(1))
        hours = int(match.group(2))
        minutes = int(match.group(3)) if match.group(3) else 0
        seconds = int(match.group(4)) if match.group(4) else 0
    elif (match := re.match(r"(?:(\d+):)?(\d+)(?::(\d+))?", time_str)):
        days = 0
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2))
        seconds = int(match.group(3)) if match.group(3) else 0
    else:
        raise ValueError(f"Invalid time format: {time_str}")

    return days * 24 + hours + minutes / 60 + seconds / 3600

def write_slurm_script(args, cli_args):
    if args.output is None:
        args.output = os.path.join(SLURM_LOG_DIR, args.job_name + ".%j.out")
    if args.error is None:
        args.error = os.path.join(SLURM_LOG_DIR, args.job_name + ".%j.err")
    args.gpus = f"gpu:{args.gpus}" if args.gpus is not None else args.gpus

    filename = os.path.join(SLURM_SCRIPT_DIR, args.job_name.replace('-', '_') + ".sbatch")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("#!/bin/bash\n")

        for arg_name in SLURM_ARGS.keys():
            arg_value = vars(args)[arg_name.replace("-", "_")]
            if arg_name in SLURM_NAME_OVERRIDES:
                arg_name = SLURM_NAME_OVERRIDES[arg_name]
            if arg_value is not None:
                f.write(f"#SBATCH --{arg_name}={arg_value}\n")

        if get_time_in_hour(args.time) > 48:
            f.write("#SBATCH --qos=long\n")

        f.write('\n')
        f.write('echo "SLURM_JOBID = "$SLURM_JOBID\n')
        f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST\n')
        f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST\n')
        f.write('echo "SLURM_NNODES = "$SLURM_NNODES\n')
        f.write('echo "SLURMTMPDIR = "$SLURMTMPDIR\n')
        f.write('echo "working directory = "$SLURM_SUBMIT_DIR\n')
        f.write('\n')
        f.write(f'source {ENV_SETUP_SCRIPT}\n')
        f.write(f'python {cli_args}\n')
        f.write('wait\n')

def write_bash_script(args, cli_args):
    filename = os.path.join(BASH_SCRIPT_DIR, args.job_name.replace('-', '_') + ".sh")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f'python {cli_args}\n')

def main():
    args, training_args = parse_args()

    if args.sweep_config:
        module = importlib.import_module("sweep_config")
        sweep_configuration = getattr(module, args.sweep_config, None)
        if sweep_configuration is None:
            raise ValueError(f"Invalid sweep configuration: {args.sweep_config}")
        args.sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project_name)

    if args.write_script is not None:
        cli_args = re.sub(r' --write_script \S+', '', ' '.join(sys.argv))
        if args.sweep_config:
            cli_args = re.sub(r'--sweep_config \S+', f'--sweep_id {args.sweep_id}', cli_args)

        if args.write_script == "slurm":
            write_slurm_script(args, cli_args)
        elif args.write_script == "bash":
            write_bash_script(args, cli_args)
    else:
        task_module = importlib.import_module(args.module.replace('/', '.'))
        importlib.reload(task_module)
        training_args = task_module.parse_args(training_args)

        if args.log_file is None:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            args.log_file = f'logs/{timestamp}.log'

        logging.basicConfig(
            filename=args.log_file,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=getattr(logging, args.log_level),
        )

        if args.sweep_id is not None:
            def sweep_function():
                wandb.init()
                sweep_args = copy.deepcopy(training_args)
                for k, v in wandb.config.items():
                    if k == "learning_rate" and isinstance(v, int):
                        v = float(v) * training_args.learning_rate
                    setattr(sweep_args, k, v)
                logger.info(
                    "Training/evaluation parameters %s", pformat(vars(sweep_args))
                )
                task_module.main(sweep_args)
            wandb.agent(
                args.sweep_id,
                function=sweep_function,
                project=args.project_name,
                count=args.job_count
            )
        else:
            if args.project_name is not None:
                wandb.init(project=args.project_name, name=args.job_name, id=args.run_id, resume="allow")
            logger.info(f"Training parameters: {pformat(vars(training_args))}")
            task_module.main(training_args)

if __name__ == "__main__":
    main()