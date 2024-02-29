#!/bin/bash

base_cmd="python -m src.posit.speech_recognition.whisper_eval --bf16 --output_dir tmp/whisper_tiny/"

# Define quantization strategies
declare -a quantize_fwds=("" "gemm" "gemm,residual" "gemm,residual,norm" "gemm,residual,norm,act" "gemm,residual,norm,act,scaling")

for quantize_fwd in "${quantize_fwds[@]}"; do
    if [ -z "$quantize_fwd" ]; then
        cmd="$base_cmd $output_dir"
    else
        cmd="$base_cmd $output_dir --quantize_weights --quantize_fwd $quantize_fwd"
    fi
    echo "Running: $cmd"
    eval $cmd
done