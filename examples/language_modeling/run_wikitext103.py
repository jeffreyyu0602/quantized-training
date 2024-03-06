import argparse
import subprocess

dtypes = ["posit8_1", "e4m3", "posit8_2"]

operations = [
    "gemm",
    "gemm,residual",
    "gemm,residual,norm",
    "gemm,residual,norm,act",
    "gemm,residual,norm,act,scaling"
]

def run_evaluation(model, dtype, ops):
    base_cmd = [
        "python", "examples/language_modeling/run_clm.py",
        "--model_name_or_path", model,
        "--dataset_name", "wikitext",
        "--dataset_config_name", "wikitext-103-raw-v1",
        "--per_device_eval_batch_size", "8",
        "--do_eval",
        "--output_dir", "tmp/llama2-clm",
        "--torch_dtype", "bfloat16",
        "--low_cpu_mem_usage",
        "--dtype", dtype,
        "--quantize_weights",
        "--quantize_fwd", ops
    ]
    print("Running:", ' '.join(base_cmd))
    subprocess.run(base_cmd, check=True)

def run_perplexity(model, dtype, ops, log_file):
    base_cmd = [
        "python", "examples/language_modeling/perplexity.py",
        "--model_id", model,
        "--max_length", "1024",
        "--dtype", dtype,
        "--quantize_fwd", ops,
        "--quantize_weights",
        "--log_file", log_file,
    ]
    print("Running:", ' '.join(base_cmd))
    subprocess.run(base_cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run language model evaluation.")
    parser.add_argument("--model_id", default="meta-llama/Llama-2-7b-hf", help="Pretrained model for evaluation.")
    parser.add_argument("--log_file", default="logs/llama2.log", help="Path to the log file.")
    args = parser.parse_args()

    for dtype in dtypes:
        for ops in operations:
            run_perplexity(args.model_id, dtype, ops, args.log_file)

    print("All commands executed.")

if __name__ == "__main__":
    main()