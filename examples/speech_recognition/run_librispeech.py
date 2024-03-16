import argparse
import subprocess

dtypes = ["posit8_1", "posit8_2", "e4m3"]

operations = [
    "gemm",
    "gemm,residual",
    "gemm,residual,norm",
    "gemm,residual,norm,act",
    "gemm,residual,norm,act,scaling"
]

def run_evaluation(model_id, dtype, ops, log_file, gpu):
    command = [
        "python",
        "examples/speech_recognition/whisper_eval.py",
        "--model_id", model_id,
        "--bf16",
        "--dtype", dtype,
        "--quantize_fwd", ops,
        "--quantize_weights",
        "--log_file", log_file,
    ]
    if gpu is not None:
        command += ['--gpu', gpu]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run Whisper evaluation with various quantization strategies.")
    parser.add_argument("--model_id", default="openai/whisper-tiny", help="Model name or path for the Whisper model")
    parser.add_argument("--log_file", default="", help="Path to the log file.")
    parser.add_argument("--gpu", default=None, help="GPU to use.")
    args = parser.parse_args()

    for dtype in dtypes:
        for ops in operations:
            run_evaluation(args.model_id, dtype, ops, args.log_file, args.gpu)

if __name__ == "__main__":
    main()