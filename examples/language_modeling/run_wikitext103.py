import argparse
import re
import subprocess

dtypes = ["posit8_1", "e4m3", "posit8_2"]

operations = [
    "gemm",
    "gemm,residual",
    "gemm,residual,norm",
    "gemm,residual,norm,act",
    "gemm,residual,norm,act,scaling"
]

def run_evaluation(model_id, max_length, stide, dtype, ops, log_file, gpu):
    command = [
        "python", "examples/language_modeling/perplexity.py",
        "--model_id", model_id,
        "--max_length", max_length,
        "--stride", stide,
        "--dtype", dtype,
        "--quantize_fwd", ops,
        "--quantize_weights",
        "--log_file", log_file,
    ]
    if gpu is not None:
        command += ['--gpu', gpu]
    print("Running:", ' '.join(command))
    subprocess.run(command, check=True)

def extract_ppl(log_file, out_file):
    with open(log_file, 'r') as file, open(out_file, 'w') as out:
        scores = (re.findall(r"perplexity: (\d+\.\d+)", file.read()))
        for score in scores:
            out.write(score + '\n')

def main():
    parser = argparse.ArgumentParser(description="Run language model evaluation.")
    parser.add_argument("--model_id", required=True, help="Pretrained model for evaluation.")
    parser.add_argument('--max_length', default=1024, help='Maximum sequence length')
    parser.add_argument('--stride', default=512, help='Stride for processing the data')
    parser.add_argument("--log_file", default="logs/ppl.log", help="Path to the log file.")
    parser.add_argument("--out_file", default=None, help="Name of the output file.")
    parser.add_argument("--gpu", default=None, help="GPU to use.")
    args = parser.parse_args()

    for dtype in dtypes:
        for ops in operations:
            run_evaluation(
                args.model_id, args.max_length, args.stride, dtype, ops, args.log_file, args.gpu
            )

            if args.out_file:
                extract_ppl(args.log_file, args.out_file)

    print("All commands executed.")

if __name__ == "__main__":
    main()