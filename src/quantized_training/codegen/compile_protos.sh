#!/bin/bash

# Check if the protoc command is available
if ! command -v protoc &> /dev/null
then
    echo "protoc could not be found, please install Protocol Buffers."
    exit 1
fi

# Absolute path to this script, regardless of where it is called from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Directory where your .proto files are stored
PROTO_DIR=$SCRIPT_DIR

# Directory where you want to output the compiled Python files
OUT_DIR=$SCRIPT_DIR

# Create the output directory if it does not exist
mkdir -p "$OUT_DIR"

# Change to the directory containing the .proto files
cd "$PROTO_DIR"

# Compile all .proto files in the current directory
for PROTO_FILE in *.proto
do
    echo "Compiling $PROTO_FILE"
    # Ensure the output path is absolute or correctly relative to the script's execution location
    protoc --python_out="$OUT_DIR" "$PROTO_FILE"
done

# Optionally, change back to the original directory if needed
# cd -

echo "Compilation complete. Output files are in $OUT_DIR"