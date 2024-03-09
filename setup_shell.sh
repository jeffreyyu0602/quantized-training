# Make sure we have the conda environment set up.
_CONDA_ROOT="${SCRATCH}/anaconda3"
# Copyright (C) 2012 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause
\. "$_CONDA_ROOT/etc/profile.d/conda.sh" || return $?

ENV_NAME=myenv
REPO_PATH="${SCRATCH}/quantized-training/"
WANDB_API_KEY="" # If you want to use wandb, set this to your API key.

# Setup Conda
conda activate $ENV_NAME

unset DISPLAY # Make sure display is not set or it will prevent scripts from running in headless mode.

if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY=$WANDB_API_KEY
fi

# First check if we have a GPU available
if nvidia-smi | grep "CUDA Version"; then
    if [ -d "/usr/local/cuda-11.3" ]; then
        export PATH=/usr/local/cuda-11.3/bin:$PATH
    elif [ -d "/usr/local/cuda-11.1" ]; then
        export PATH=/usr/local/cuda-11.1/bin:$PATH
    elif [ -d "/usr/local/cuda-11.0" ]; then
        export PATH=/usr/local/cuda-11.0/bin:$PATH
    elif [ -d "/usr/local/cuda-10.2" ]; then
        export PATH=/usr/local/cuda-10.2/bin:$PATH
    elif [ -d "/usr/local/cuda" ]; then
        export PATH=/usr/local/cuda/bin:$PATH
        echo "Using default CUDA. Compatibility should be verified."
    else
        echo "Warning: Could not find a CUDA version but GPU was found."
    fi
    # Setup any GPU specific flags
else
    echo "GPU was not found, assuming CPU setup."
fi
