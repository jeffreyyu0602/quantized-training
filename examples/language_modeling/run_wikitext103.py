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

def run_evaluation(model_id, max_length, stide, dtype, ops, log_file):
    base_cmd = [
        "python", "examples/language_modeling/perplexity.py",
        "--model_id", model_id,
        "--max_length", str(max_length),
        "--stride", str(stide),
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
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--stride', type=int, default=512, help='Stride for processing the data')
    parser.add_argument("--log_file", default="logs/llama2.log", help="Path to the log file.")
    args = parser.parse_args()

    for dtype in dtypes:
        for ops in operations:
            run_evaluation(args.model_id, args.max_length, args.stride, dtype, ops, args.log_file)

    print("All commands executed.")

if __name__ == "__main__":
    main()