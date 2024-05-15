#!/bin/bash

# Check if the protoc command is available
if ! command -v protoc &> /dev/null
then
    echo "protoc could not be found, please install Protocol Buffers."
    exit 1
fi

# Directory where your .proto files are stored
PROTO_DIR="./src/quantized_training/accelerator/"  # Adjust this path to where your .proto files are located.

# Directory where you want to output the compiled Python files
OUT_DIR="./src/quantized_training/accelerator/build" # Adjust this to your preferred output directory.

# Create the output directory if it does not exist
mkdir -p "$OUT_DIR"

# Compile all .proto files in the PROTO_DIR
for PROTO_FILE in "$PROTO_DIR"/*.proto
do
    echo "Compiling $PROTO_FILE"
    protoc --python_out="$OUT_DIR" "$PROTO_FILE"
done

echo "Compilation complete. Output files are in $OUT_DIR"