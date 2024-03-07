import argparse
import os
import subprocess

operations = [
    None,
    "gemm",
    "gemm,residual",
    "gemm,residual,norm",
    "gemm,residual,norm,act",
    "gemm,residual,norm,act,scaling"
]

def main():
    parser = argparse.ArgumentParser(description="Run Whisper evaluation with various quantization strategies.")
    parser.add_argument("--model_id", default="openai/whisper-tiny", help="Model name or path for the Whisper model")
    parser.add_argument("--output_dir", default="tmp/whisper/", help="Output directory for generated text and plots")
    parser.add_argument('--out_file', default='accuracy.out')
    parser.add_argument('--gpu', default=None)
    args = parser.parse_args()

    base_cmd = ["python", "examples/speech_recognition/whisper_eval.py", "--model_id", args.model_id, "--bf16"]

    for ops in operations:
        output_dir = "bf16" if ops is None else ops.replace(",", "_")
        cmd = base_cmd + ["--output_dir", os.path.join(args.output_dir, output_dir)]
        if ops is not None:
            cmd += ["--quantize_weights", "--quantize_fwd", ops]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()