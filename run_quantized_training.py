import argparse
import os
import re
import subprocess
from datetime import datetime


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
    args, training_args = parser.parse_known_args()

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

    return args, training_args

def get_base_cmd(args):
    cmd_parts = ["python task_runner.py"]

    if args.task in {"mnli", "qnli", "mrpc", "sst2"}:
        cmd_parts.append(f"--module examples/text_classification/run_glue_no_trainer.py --task_name {args.task} --max_length 128")
    elif args.task == "squad":
        cmd_parts.append(f"--module examples/question_answering/run_qa_no_trainer.py --dataset_name {args.task} --max_seq_length 384")
    else:
        raise ValueError(f"Invalid task: {args.task}")

    cmd_parts.extend([
        "--pad_to_max_length",
        f"--model_name_or_path {args.model}",
        f"--per_device_train_batch_size {args.batch_size}",
        f"--per_device_eval_batch_size 16",
        f"--learning_rate {args.learning_rate}",
        f"--num_train_epochs {args.num_train_epochs}",
        f"--seed {args.seed}",
        f"--checkpointing_steps epoch",
        f"--lora_rank {args.lora_rank}",
        f"--lora_alpha {args.lora_alpha}",
        f"--target_modules {args.target_modules}",
        f"--bf16",
        f"--do_train",
    ])

    if args.model == "models/mobilebert_tiny":
        cmd_parts.append("--num_hidden_layers 21")

    if args.model == "roberta-large-mnli":
        cmd_parts.append("--ignore_mismatched_sizes")

    return ' '.join(cmd_parts)

def main():
    args, training_args = parse_args()

    if args.resume:
        job_prefix = f"{MODEL_NAME_MAPPING.get(args.model, args.model)}-{args.task}"
        names = ["bf16", "posit8", "posit8-approx", "posit8-approx-shifted", "fp8"]
        for name in names:
            filename = f"{job_prefix}-lora-{name}.sbatch".replace('-', '_')
            script_path = os.path.join("slurm_scripts", filename)
            with open(script_path, "r") as f:
                lines = f.readlines()

            if (match := re.search(r'--output_dir (\S+)', lines[-2])) is not None:
                output_dir = match.group(1)
            else:
                raise ValueError("Could not find output_dir argument in command:", lines[-2])

            if not os.path.isdir(output_dir):
                raise ValueError(f"Checkpoint directory does not exist: {output_dir}")

            model_dirs = [d for d in os.listdir(output_dir) if d.startswith("step_")]
            model_dirs.sort(key=lambda x: int(x[5:]))
            checkpoint_path = os.path.join(output_dir, model_dirs[-1]) if model_dirs else None

            if 'resume_from_checkpoint' in lines[-2]:
                lines[-2] = re.sub(r'(--resume_from_checkpoint )\S+', r'\1' + checkpoint_path, lines[-2])
            else:
                lines[-2] = lines[-2].rstrip('\n') + f" --resume_from_checkpoint {checkpoint_path}\n"

            # TODO: load checkpoint and read run_id from it
            # lines[-2] = lines[-2].rstrip('\n') + f" --run_id {run_id}\n"

            with open(script_path, "w") as f:
                f.writelines(lines)
    else:
        base_cmd = get_base_cmd(args)
        quant_args = f" --quantize_weights --quantize_fwd {args.quantized_ops} --quantize_bwd {args.quantized_ops} --scaling_bwd"
        posit_args = f" --dtype posit8_1 --max_fwd 64 --max_bwd 64"
        fp8_args = f" --dtype FP8 --max_fwd 448 --max_bwd 57344"

        commands = {
            "bf16": base_cmd,
            "posit8": base_cmd + quant_args + posit_args,
            "posit8-approx": base_cmd + quant_args + posit_args + f" --posit_reciprocal --posit_exp",
            "posit8-approx-shifted": base_cmd + quant_args + posit_args + " --posit_reciprocal --posit_exp_shifted",
            "fp8": base_cmd + quant_args + fp8_args,
        }

        job_prefix = f"{MODEL_NAME_MAPPING.get(args.model, args.model)}-{args.task}"
        for name, cmd_str in commands.items():
            # TODO: new slurm script will overwrite old one. Using datetime as folder
            # name will lose track of the checkpoint directory. Move existing scripts
            # to a new folder and move it back when resuming from previous checkpoint.
            if args.save_ckpt:
                output_dir = datetime.now().strftime("run-%Y%m%d_%H%M%S")
                cmd_str += f' --output_dir models/{output_dir}'

            if training_args:
                cmd_str += ' ' + ' '.join(training_args)

            job_name = f"{job_prefix}-lora-{name}"
            cmd_str += f" --project_name {job_prefix}-quantized-training --job-name {job_name} --write_script slurm"

            if args.submit_jobs:
                subprocess.run(cmd_str, shell=True, text=True, check=True)
                script_path = os.path.join("slurm_scripts", f"{job_name.replace('-', '_')}.sbatch")
                subprocess.run(f"sbatch {script_path}", shell=True, text=True, check=True)

            print("\n" + cmd_str)

if __name__ == "__main__":
    main()