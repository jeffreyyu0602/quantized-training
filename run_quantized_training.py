import argparse
import os
import re
import subprocess
from datetime import datetime

import torch


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
        "quantized_ops": "gemm,residual,norm,act"
    },
    "roberta-large": {
        "lora_rank": 8,
        "lora_alpha": 16,
        "target_modules": "query,value",
        "quantized_ops": "gemm,residual,norm,act"
    },
    "roberta-large-mnli": {
        "lora_rank": 8,
        "lora_alpha": 16,
        "target_modules": "query,value",
        "quantized_ops": "gemm,residual,norm,act"
    },
}

MODEL_NAME_MAPPING = {
    "models/mobilebert_tiny": "mobilebert-tiny",
    "google/mobilebert-uncased": "mobilebert",
    "roberta-large-mnli": "roberta-large",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate training commands with different quantization data types.")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("-bs", "--batch_size", type=int, required=True)
    parser.add_argument("-lr", "--learning_rate", type=float, required=True)
    parser.add_argument("-epochs", "--num_train_epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank to use")
    parser.add_argument("--lora_alpha", type=int, default=None, help="Scaling factor for LoRA")
    parser.add_argument("--target_modules", type=str, default=None, help="LoRA layers")
    parser.add_argument("--quantized_ops", type=str, default=None, help="Quantized ops to use")
    parser.add_argument("--save_ckpt", action="store_true", help="Whether to save model")
    parser.add_argument("--submit_jobs", action="store_true", help="Submit slurm job.")
    parser.add_argument("--resume", action="store_true", help="Resume training from stored checkpoint")
    args, extra_args = parser.parse_known_args()

    # Remove trailing slash from model path
    args.model = args.model.rstrip('/')

    if not args.resume and (args.batch_size is None or args.learning_rate is None or args.num_train_epochs is None):
        raise ValueError("Must specify batch size, learning rate, and number of epochs if not resuming from a previous checkpoint")

    if (config := LORA_CONFIG.get(args.model)):
        for k, v in config.items():
            if getattr(args, k) is None:
                setattr(args, k, v)
    else:
        config_keys = ["lora_rank", "lora_alpha", "target_modules", "quantized_ops"]
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
            filename = os.path.join("slurm_scripts", f"{prefix}-lora-{name}.sbatch".replace('-', '_'))
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

        quant_args = [
            '--quantize_fwd', args.quantized_ops,
            '--quantize_bwd', args.quantized_ops,
            '--quantize_weights',
            '--scaling_bwd'
        ]
        posit_args = ['--dtype', 'posit8_1', '--max_fwd', '64', '--max_bwd', '64']
        fp8_args = ['--dtype', 'FP8', '--max_fwd', '448', '--max_bwd', '57344']

        commands = {
            "bf16": base_cmd,
            "posit8": base_cmd + quant_args + posit_args,
            "posit8-approx": base_cmd + quant_args + posit_args + ["--posit_reciprocal", "--posit_exp"],
            "posit8-approx-shifted": base_cmd + quant_args + posit_args + ["--posit_reciprocal", "--posit_exp_shifted"],
            "fp8": base_cmd + quant_args + fp8_args,
        }

        prefix = f"{MODEL_NAME_MAPPING.get(args.model, args.model)}-{args.task}"
        for name, command in commands.items():
            job_name = f"{prefix}-lora-{name}"

            if args.save_ckpt:
                output_dir = datetime.now().strftime("run-%Y%m%d_%H%M%S")
                command += ['--output_dir', f'models/{output_dir}']

            if extra_args:
                command += extra_args

            command += [
                '--project', f'{prefix}-quantized-training', '--run_name', job_name,
                'slurm', '--job_name', job_name,
            ]

            print("Running:", ' '.join(command))
            if args.submit_jobs:
                filename = os.path.join("slurm_scripts", f"{job_name}.sbatch")
                subprocess.run(command, check=True)
                subprocess.run(f"sbatch {filename}", check=True)

if __name__ == "__main__":
    main()