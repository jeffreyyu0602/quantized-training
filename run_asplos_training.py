import argparse
import os
import re
import subprocess
from datetime import datetime

import torch

HYPERPARAMETERS = {
    'models/mobilebert_tiny': {
        'mnli': [16, 12e-4, 30],
        'qnli': [16, 8e-4, 50],
        'mrpc': [16, 11e-4, 30],
        'sst2': [16, 10e-4, 60],
        'squad': [16, 10e-2, 30],
    },
    'google/mobilebert-uncased': {
        'mnli': [16, 12e-4, 30],
        'qnli': [16, 8e-4, 50],
        'mrpc': [16, 8e-4, 30],
        'sst2': [16, 8e-4, 60],
        'squad': [16, 10e-2, 30],
    },
    'roberta-base': {
        'mnli': [16, 14e-4, 30],
        'qnli': [32, 7e-4, 25],
        'mrpc': [16, 5e-4, 50],
        'sst2': [16, 9e-4, 60],
        'squad': [16, 10e-4, 30]
    },
    'roberta-large': {
        'mnli': [4, 7e-4, 10],
        'qnli': [4, 4e-4, 10],
        'mrpc': [4, 5e-4, 20],
        'sst2': [4, 5e-4, 10],
        'squad': [4, 5e-4, 10]
    },
    'roberta-large-mnli': {
        'mrpc': [4, 5e-4, 20],
    },
}

LORA_CONFIG = {
    "models/mobilebert_tiny": {
        "lora_rank": 8,
        "lora_alpha": 8,
        "target_modules": "query,key,value,dense",
        "quantized_ops": "gemm"
    },
    "google/mobilebert-uncased": {
        "lora_rank": 8,
        "lora_alpha": 8,
        "target_modules": "query,key,value,dense",
        "quantized_ops": "gemm"
    },
    "roberta-base": {
        "lora_rank": 8,
        "lora_alpha": 8,
        "target_modules": "query,value",
        "quantized_ops": "gemm,residual,layernorm,activation"
    },
    "roberta-large": {
        "lora_rank": 8,
        "lora_alpha": 16,
        "target_modules": "query,value",
        "quantized_ops": "gemm,residual,layernorm,activation"
    },
    "roberta-large-mnli": {
        "lora_rank": 8,
        "lora_alpha": 16,
        "target_modules": "query,value",
        "quantized_ops": "gemm,residual,layernorm,activation"
    },
}

MODEL_NAME_MAPPING = {
    "models/mobilebert_tiny": "mobilebert-tiny",
    "google/mobilebert-uncased": "mobilebert",
    "roberta-large-mnli": "roberta-large",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate training commands with different quantization data types.")
    parser.add_argument("--task", required=True, help="GLUE or SQuAD task to run.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("-bs", "--batch_size", type=int, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    parser.add_argument("-epochs", "--num_train_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank to use")
    parser.add_argument("--lora_alpha", type=int, default=None, help="Scaling factor for LoRA")
    parser.add_argument("--target_modules", type=str, default=None, help="LoRA layers")
    parser.add_argument("--quantized_ops", type=str, default=None, help="Quantized ops to use")
    parser.add_argument("--save_ckpt", action="store_true", help="Whether to save model")
    parser.add_argument("--resume", action="store_true", help="Resume training from stored checkpoint")
    parser.add_argument("--wandb_log", action="store_true", help="Setup W&B logging.")
    parser.add_argument("--slurm", action="store_true", help="Submit slurm job.")
    parser.add_argument("--run_job", nargs="?", const="all", default="", help="Run generated command.")
    args, extra_args = parser.parse_known_args()

    # Remove trailing slash from model path
    args.model = args.model.rstrip('/')

    params = {}
    if (configs := HYPERPARAMETERS.get(args.model)) and (config := configs.get(args.task)):
        params = {
            "batch_size": config[0],
            "learning_rate": config[1],
            "num_train_epochs": config[2]
        }

    if (config := LORA_CONFIG.get(args.model)):
        params = {**params, **config}

    for k, v in params.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    config_keys = [
        "batch_size",
        "learning_rate",
        "num_train_epochs",
        "lora_rank",
        "lora_alpha",
        "target_modules",
        "quantized_ops",
    ]
    missing_keys = [key for key in config_keys if getattr(args, key) is None]
    assert not missing_keys, (
        f"The following keys must be specified for {args.model}: {', '.join(missing_keys)}"
    )

    return args, extra_args

def get_base_cmd(args):
    command = ["python"]
    if args.task in {"mnli", "qnli", "mrpc", "sst2"}:
        command += [
            "examples/text_classification/run_glue_no_trainer.py",
            "--task_name", args.task,
            "--max_length", "128"
        ]
    elif args.task == "squad":
        command += [
            "examples/question_answering/run_qa_no_trainer.py",
            "--dataset_name", args.task,
            "--max_seq_length", "384"
        ]
    else:
        raise ValueError(f"Invalid task: {args.task}")

    command += [
        '--pad_to_max_length',
        '--model_name_or_path', args.model,
        '--per_device_train_batch_size', str(args.batch_size),
        '--per_device_eval_batch_size', str(args.batch_size),
        '--learning_rate', str(args.learning_rate),
        '--num_train_epochs', str(args.num_train_epochs),
        '--seed', str(args.seed),
        '--checkpointing_steps', 'epoch',
        '--lora_rank', str(args.lora_rank),
        '--lora_alpha', str(args.lora_alpha),
        '--target_modules', args.target_modules,
        '--warmup_ratio', '0.06',
        '--bf16',
        '--do_train',
    ]

    if args.model == "models/mobilebert_tiny":
        command += ["--num_hidden_layers", "21"]

    if args.model == "roberta-large-mnli":
        command += ["--ignore_mismatched_sizes"]

    return command

def main():
    args, extra_args = parse_args()

    if args.resume:
        prefix = f"{MODEL_NAME_MAPPING.get(args.model, args.model)}-{args.task}"
        for name in ["bf16", "posit8", "posit8-approx", "posit8-approx-shifted", "fp8"]:
            filename = os.path.join("slurm_scripts", f"{prefix}-{name}-{args.seed}.sbatch")
            with open(filename, "r") as f:
                content = f.read()

            if (match := re.search(r'--output_dir (\S+)', content)) is not None:
                output_dir = match.group(1)
            else:
                raise ValueError("Could not find output_dir argument in command:")

            assert os.path.isdir(output_dir), (
                f"Checkpoint directory does not exist: {output_dir}"
            )

            model_dirs = [d for d in os.listdir(output_dir) if d.startswith("step_")]
            model_dirs.sort(key=lambda x: int(x[5:]))
            assert model_dirs, f"No checkpoints found in {output_dir}"

            checkpoint_path = os.path.join(output_dir, model_dirs[-1])
            if 'resume_from_checkpoint' in content:
                content = re.sub(r'(--resume_from_checkpoint )\S+', r'\1' + checkpoint_path, content)
            else:
                content = re.sub(r'(--do_train)', r'\1' + f" --resume_from_checkpoint " + checkpoint_path, content)

            checkpoint = torch.load(os.path.join(output_dir, "checkpoint.tar"))
            content = re.sub(r'(--run_name \S+)', r'\1' + " --run_id " + checkpoint['run_id'], content)

            with open(filename, "w") as f:
                f.write(content)
    else:
        base_cmd = get_base_cmd(args)
        quant_args = ['--quantize_forward', args.quantized_ops, '--quantize_backprop', args.quantized_ops]
        posit_args = [
            '--activation', 'posit8_1',
            '--weight', 'posit8_1',
            '--error', 'posit8_1,qs=per_tensor_symmetric,qmax=64,ahl=10',
        ]
        fp8_args = [
            '--activation', 'fp8_e4m3',
            '--weight', 'fp8_e4m3',
            '--error', 'fp8_e5m2,qs=per_tensor_symmetric,qmax=57344,ahl=10',
        ]

        commands = {
            "bf16": base_cmd,
            "posit8": base_cmd + quant_args + posit_args,
            "posit8-approx": base_cmd + quant_args + posit_args + [
                "--posit_reciprocal", "--posit_exp"
            ],
            "posit8-approx-shifted": base_cmd + quant_args + posit_args + [
                "--posit_reciprocal", "--posit_exp_shifted"
            ],
            "fp8": base_cmd + quant_args + fp8_args,
        }

        prefix = f"{MODEL_NAME_MAPPING.get(args.model, args.model)}-{args.task}"
        for name, command in commands.items():
            job_name = f"{prefix}-{name}-{args.seed}"

            if args.save_ckpt:
                output_dir = datetime.now().strftime("run-%Y%m%d_%H%M%S")
                command += ['--output_dir', f'models/{output_dir}']

            if extra_args:
                command += extra_args

            if args.wandb_log:
                command += [
                    '--project', f'{prefix}-quantized-training',
                    '--run_name', f'{name}-{args.seed}'
                ]

            if args.slurm:
                command += ['slurm', '--job-name', job_name]

            if args.run_job == "all" or name in args.run_job.split(","):
                print("Running:", ' '.join(command), "\n")
                subprocess.run(command, check=True)
                if args.slurm:
                    filename = os.path.join("slurm_scripts", f"{job_name}.sbatch")
                    subprocess.run(["sbatch", filename], check=True)

if __name__ == "__main__":
    main()
