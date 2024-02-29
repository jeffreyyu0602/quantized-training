#!/bin/bash

model_path="meta-llama/Llama-2-7b-hf" # Default model path

# Using getopts for single-letter options
while getopts "m:" opt; do
  case $opt in
    m) model_path="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done

# Define the base command parameters
base_cmd="python -m src.posit.language_modeling.run_clm"
dataset_name="wikitext"
dataset_config="wikitext-103-raw-v1"
batch_size="8"
output_dir="tmp/llama2-clm"
torch_dtype="bfloat16"
low_cpu="true"

# Define data types
declare -a dtypes=("posit8_1" "e4m3" "posit8_2")

# Define quantization strategies
declare -a quantize_fwds=("gemm" "gemm,residual" "gemm,residual,norm" "gemm,residual,norm,act" "gemm,residual,norm,act,scaling")

for dtype in "${dtypes[@]}"; do
    for quantize_fwd in "${quantize_fwds[@]}"; do
        cmd="$base_cmd --model_name_or_path $model_path \
--dataset_name $dataset_name \
--dataset_config_name $dataset_config \
--per_device_eval_batch_size $batch_size \
--do_eval \
--output_dir $output_dir \
--torch_dtype $torch_dtype \
--low_cpu_mem_usage \
--dtype $dtype \
--quantize_weights \
--quantize_fwd $quantize_fwd"
        echo "Running: $cmd"
        eval $cmd
    done
done

echo "All commands executed."