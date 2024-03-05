import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run Whisper evaluation with various quantization strategies.")
    parser.add_argument("--model_name_or_path", default="openai/whisper-tiny", help="Model name or path for the Whisper model")
    parser.add_argument("--output_dir", default="tmp/whisper/", help="Output directory for generated text and plots")
    args = parser.parse_args()

    base_cmd = ["python", "examples/speech_recognition/whisper_eval.py", "--model_name_or_path", args.model_name_or_path, "--bf16"]
    quantize_fwds = [None, "gemm", "gemm,residual", "gemm,residual,norm", "gemm,residual,norm,act", "gemm,residual,norm,act,scaling"]

    for quantize_fwd in quantize_fwds:
        output_dir = "bf16" if quantize_fwd is None else quantize_fwd.replace(",", "_")
        cmd = base_cmd + ["--output_dir", os.path.join(args.output_dir, output_dir)]
        if quantize_fwd is not None:
            cmd += ["--quantize_weights", "--quantize_fwd", quantize_fwd]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()